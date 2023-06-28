#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import numpy as np
import os
import boto3
from datetime import datetime
from sagemaker import get_execution_role, session


# In[112]:


# Initialize sagemaker session
sagemaker_session = session.Session()

region = sagemaker_session.boto_region_name
print(f"Region: {region}")

role = get_execution_role()
print(f"Role: {role}")

bucket = sagemaker_session.default_bucket()

prefix = "sagemaker/DEMO-sagemaker-clarify"


# In[113]:


bucket = 'syntheticdata11/data'
data_key = 'forecast.csv'
data_location = 's3://{}/{}/'.format(bucket, data_key)
df = pd.read_csv(data_location)
df


# In[114]:


print(df.loc[0])


# In[115]:


df.isnull().sum()


# In[116]:


display(df.describe)
hist = df.hist(bins=30, sharey = True, figsize=(20,10))


# In[117]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap="YlGnBu",annot=True,fmt=".1f")
plt.show()


# In[118]:


df.columns


# In[119]:


#feature extraction
sns.set_style('whitegrid')
for cols in ['temperature_2m','winddirection_10m']:
    sns.lmplot(x ='windspeed_10m', y =cols, data = df)


# In[120]:


import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session
import numpy as np


# In[121]:


# Split the data into training and testing set
X=df[['temperature_2m','winddirection_10m']]
print(X)
y=df['windspeed_10m']
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_display = df.loc[X_train.index]
X_test_display = df.loc[X_test.index]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[122]:


train = pd.concat([pd.Series(y_train, index = X_train.index, name = 'windspeed_10m', dtype = float), X_train], axis = 1).to_csv('train.csv', index=False, header=False)
test = pd.concat([pd.Series(y_test, index = X_test.index, name = 'windspeed_10m', dtype = float), X_test], axis = 1).to_csv('test.csv', index=False, header=False)


# In[123]:


import os
bucket = sagemaker.Session().default_bucket()
prefix = 'sagemake-linearreg-london-weather-prediction'
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'syntheticdata11/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'syntheticdata11/test.csv')).upload_file('test.csv')


# In[124]:


# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
prefix = 'sagemake-linearreg-london-weather-prediction/syntheticdata11'
s3_output_location = 's3://{}/{}/'.format(bucket, prefix, 'linear-learner')
from sagemaker import image_uris

container = image_uris.retrieve(region=boto3.Session().region_name, framework="linear-learner")
deploy_amt_model = True
print(container)


# In[128]:


import boto3
import sagemaker

sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(
    container,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.c4.xlarge",
    output_path=s3_output_location,
    sagemaker_session=sess
)
linear.set_hyperparameters(feature_dim=2, predictor_type="binary_classifier", mini_batch_size=20)


# In[129]:


s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data="s3://{}/{}/train".format(bucket, prefix),
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None
)

s3_input_test = sagemaker.inputs.TrainingInput(
    s3_data="s3://{}/{}/test".format(bucket, prefix),
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None
)

linear.fit({"train": s3_input_train})


# In[56]:


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


# In[ ]:


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


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

x = predictions
y = actuals

plt.title("lmplot")
plt.plot(x, y, color="red")

plt.show()

