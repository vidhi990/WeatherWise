#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd


# # LONDON WEATHER PREDICTION MODEL

# In[139]:


bucket = 'modeldata11/datafile'
data_key = 'london_weather.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)
df = pd.read_csv(data_location)
df


# In[140]:


print(df.loc[0])


# In[141]:


print(len(df))


# In[142]:


df.isnull().sum()


# As we can see that the data is cleaned, there are no null values in the data.

# In[143]:


df.dtypes


# In[144]:


temp = []
for i in range(len(df["time"])):
  temp.append(str(df["time"][i]).split("T")[0])
df["date"] = temp
df.head(2)


# In[145]:


temp = []
for i in range(len(df["time"])):
  temp.append(str(df["time"][i]).split("T")[-1])
df["seperated_time_column"] = temp
df.head(10)


# In[146]:


# Checking the size of dataset.
df.shape


# In[147]:


df.nunique()


# In[148]:


display(df.describe)
hist = df.hist(bins=30, sharey = True, figsize=(20,10))


# In[149]:


df.corr()


# In[150]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap="YlGnBu",annot=True,fmt=".1f")
plt.show()


# In[151]:


#feature extraction
sns.set_style('whitegrid')
for cols in ['temperature','windDirection']:
    sns.lmplot(x ='windSpeed', y =cols, data = df)


# In[152]:


sns.boxplot(x ='date', y ='windSpeed', data = df)


# In[153]:


import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session
import numpy as np


# In[154]:


# Split the data into training and testing set
X=df[['temperature','windDirection']]
y=df['windSpeed']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_display = df.loc[X_train.index]
X_test_display = df.loc[X_test.index]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[155]:


train = pd.concat([pd.Series(y_train, index = X_train.index, name = 'windSpeed', dtype = float), X_train], axis = 1).to_csv('train.csv', index=False, header=False)
test = pd.concat([pd.Series(y_test, index = X_test.index, name = 'windSpeed', dtype = float), X_test], axis = 1).to_csv('test.csv', index=False, header=False)


# In[156]:


import os
bucket = sagemaker.Session().default_bucket()
prefix = 'sagemake-ann-london-weather-prediction'
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'modeldata11/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'modeldata11/test.csv')).upload_file('test.csv')


# In[157]:


# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
prefix = 'sagemake-ann-london-weather-prediction/modeldata11'
s3_output_location = 's3://{}/{}/'.format(bucket, prefix, 'linear-learner')
from sagemaker import image_uris

container = image_uris.retrieve(region=boto3.Session().region_name, framework="linear-learner")
deploy_amt_model = True
print(container)


# In[158]:


import boto3
import sagemaker

sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(
    container,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.c4.xlarge",
    output_path=s3_output_location,
    sagemaker_session=sess,
    content_type = 'text/csv'
)
linear.set_hyperparameters(feature_dim=3, predictor_type="binary_classifier", mini_batch_size=20)


# In[159]:


s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data="s3://{}/{}/train".format(bucket, prefix),
    distribution="FullyReplicated",
    content_type="text/csv",
    record_wrapping=None,
    compression=None
)

s3_input_test = sagemaker.inputs.TrainingInput(
    s3_data="s3://{}/{}/test".format(bucket, prefix),
    distribution="FullyReplicated",
    content_type="text/csv",
    record_wrapping=None,
    compression=None
)

linear.fit({"train": s3_input_train})


# In[130]:


import time
from sagemaker.tuner import IntegerParameter, ContinuousParameter
from sagemaker.tuner import HyperparameterTuner

job_name = "DEMO-ll-mni-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print("Tuning job name:", job_name)

# Linear Learner tunable hyper parameters can be found here https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner-tuning.html
hyperparameter_ranges = {
    "wd": ContinuousParameter(1e-7, 1, scaling_type="Auto"),
    "learning_rate": ContinuousParameter(1e-5, 1, scaling_type="Auto"),
    "mini_batch_size": IntegerParameter(100, 2000, scaling_type="Auto"),
}

# Increase the total number of training jobs run by AMT, for increased accuracy (and training time).
max_jobs = 6
# Change parallel training jobs run by AMT to reduce total training time, constrained by your account limits.
# if max_jobs=max_parallel_jobs then Bayesian search turns to Random.
max_parallel_jobs = 2


hp_tuner = HyperparameterTuner(
    linear,
    "validation:binary_f_beta",
    hyperparameter_ranges,
    max_jobs=max_jobs,
    max_parallel_jobs=max_parallel_jobs,
    objective_type="Maximize",
)


# Launch a SageMaker Tuning job to search for the best hyperparameters
hp_tuner.fit(inputs={"train": s3_input_train, "test": s3_input_test}, job_name=job_name)


# In[134]:


from datetime import datetime

endpoint_name = f"DEMO-{datetime.utcnow():%Y-%m-%d-%H%M}"
print("EndpointName =", endpoint_name)


# In[137]:


if deploy_amt_model:
    linear_predictor = hp_tuner.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
else:
    linear_predictor = linear.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")


# In[ ]:


from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

linear_predictor.serializer = CSVSerializer()
linear_predictor.deserializer = JSONDeserializer()


# In[ ]:


result = linear_predictor.predict(train_set[0][30:31], initial_args={"ContentType": "text/csv"})
print(result)


# In[ ]:


import numpy as np

predictions = []
for array in np.array_split(test_set[0], 100):
    result = linear_predictor.predict(array)
    predictions += [r["predicted_label"] for r in result["predictions"]]

predictions = np.array(predictions)


# In[ ]:


import pandas as pd

pd.crosstab(
    np.where(test_set[1] == 0, 1, 0), predictions, rownames=["actuals"], colnames=["predictions"]
)


# In[ ]:




