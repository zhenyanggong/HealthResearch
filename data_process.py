#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Zhenyang Gong
# Data Challenge
import os
import pandas as pd
import feather
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
import itertools
import string

# helper function of reading input dataset
def read_feature(cwd, file_name):
	data_file_path = cwd + "/data/" + file_name
	return feather.read_dataframe(data_file_path)


# In[2]:



cwd = os.getcwd()
allergy_data = read_feature(cwd, "allergies.feather")


# In[ ]:


condition_data = read_feature(cwd, "conditions.feather")


# In[ ]:


immunization_data = read_feature(cwd, "immunizations.feather")


# In[ ]:


observation_data = read_feature(cwd, "observations.feather")


# In[ ]:


procedure_data = read_feature(cwd, "procedures.feather")


# In[ ]:


careplan_data = read_feature(cwd, "careplans.feather")


# In[ ]:


encounter_data = read_feature(cwd, "encounters.feather")


# In[ ]:


medication_data = read_feature(cwd, "medications.feather")


# In[ ]:


patient_data = read_feature(cwd, "patients.feather")


# In[ ]:


# Create a sample of patients with emergency visits between from 2008-2016
# Select from the encounter dataset by whether contain "emergercy" in the description column. (case insensitive)
emergency_data = encounter_data[encounter_data['DESCRIPTION'].str.contains('Emergency',regex=False, case=False, na=False)]
emergency_data = emergency_data[(emergency_data['DATE'] > '2008-1-1') & (emergency_data['DATE'] <= '2016-12-20')]
# emergency_data


# In[ ]:



patient_data = patient_data.rename(columns={'ID': 'PATIENT'})
df = pd.merge(emergency_data, patient_data, on='PATIENT')
# clean up the dataset a bit 
df.drop(['CODE', 'DESCRIPTION', 'REASONCODE', 'REASONDESCRIPTION', 'SSN', 'DRIVERS', 'PASSPORT', 'FIRST', 'LAST', 'MAIDEN'], axis=1, inplace=True) 
# df


# In[ ]:


# 848 visits from above
# summary of age
# following summary on demographics is based on number of visits intead of unique patients !!!
df.describe().transpose()


# In[ ]:


# summary of gender
df['GENDER'].value_counts()


# In[ ]:


# summary of race
df['RACE'].value_counts()


# In[ ]:


# unique patient number
df['PATIENT'].nunique()


# In[ ]:


# merge with condition dataset to include medical conditions
condition_df = pd.merge(df, condition_data, on='PATIENT')
# change dtype to datetime64[ns]
condition_df['DATE'] = pd.to_datetime(condition_df['DATE'])
condition_df['ONE_YEAR_AGO'] = condition_df['DATE'] - pd.DateOffset(years=1)
condition_df['ONE_YEAR_AGO'] = condition_df['ONE_YEAR_AGO'].astype(str)
condition_df['DATE'] = condition_df['DATE'].astype(str)
# select rows that have medical conditions diagnosed over the year prior to the emergency visits
# a patient can have multiple conditions diagnosed within the year !!!
# if df.START == df.DATE, it is the medical conditions disgnosed during the emergency vist, so use df.START < df.DATE
condition_df = condition_df[(condition_df.START >= condition_df.ONE_YEAR_AGO) & (condition_df.START < condition_df.DATE)]


# In[ ]:



# condition_df['DESCRIPTION'].value_counts()


# In[ ]:



regression_data = df[['DATE','PATIENT', 'DEATHDATE', 'RACE', 'ADDRESS']]
regression_data['DATE'] = pd.to_datetime(regression_data['DATE'])
regression_data['ONE_YEAR_LATER'] = regression_data['DATE'] + pd.DateOffset(years=1)
regression_data['ONE_YEAR_LATER'] = regression_data['ONE_YEAR_LATER'].astype(str)
regression_data['MORTALITY'] = 0
regression_data.loc[((regression_data.DEATHDATE != 'None') & (regression_data.DEATHDATE < regression_data.ONE_YEAR_LATER)), 'MORTALITY'] = 1
regression_data['MORTALITY'].value_counts()


# In[ ]:


# create three dummies for race and take white as the baseline
regression_data['ASIAN_DUMMY'] = 0
regression_data.loc[regression_data.RACE == 'asian', 'ASIAN_DUMMY'] = 1
regression_data['HISPANIC_DUMMY'] = 0
regression_data.loc[regression_data.RACE == 'hispanic', 'HISPANIC_DUMMY'] = 1
regression_data['BLACK_DUMMY'] = 0
regression_data.loc[((regression_data.RACE == 'black') | (regression_data.RACE == 'black or african american')), 'BLACK_DUMMY'] = 1
# regression_data


# In[ ]:


# estimate income by using ACS data
# read in 2012-2016 ACS dataset
acs_directory = cwd + '/ACS'
acs_12 = pd.read_csv(acs_directory + '/ACS_12.csv')
acs_12['YEAR'] = '2012'
acs_13 = pd.read_csv(acs_directory + '/ACS_13.csv')
acs_13['YEAR'] = '2013'
acs_14 = pd.read_csv(acs_directory + '/ACS_14.csv')
acs_14['YEAR'] = '2014'
acs_15 = pd.read_csv(acs_directory + '/ACS_15.csv')
acs_15['YEAR'] = '2015'
acs_16 = pd.read_csv(acs_directory + '/ACS_16.csv')
acs_16['YEAR'] = '2016'
# acs_16


# In[ ]:


# clean up acs data and merge them into a big ACS table only include "YEAR", "ZIP CODE", "MEDIAN HOUSEHOLD INCOM"
acs_12 = acs_12.rename(columns={'GEO.id2': 'ZIP', 'HC02_EST_VC02': 'INCOME'})
acs_12 = acs_12[['ZIP', 'YEAR', 'INCOME']]
acs_13 = acs_13.rename(columns={'GEO.id2': 'ZIP', 'HC01_EST_VC02': 'INCOME'})
acs_13 = acs_13[['ZIP', 'YEAR', 'INCOME']]
acs_14 = acs_14.rename(columns={'GEO.id2': 'ZIP', 'HC02_EST_VC02': 'INCOME'})
acs_14 = acs_14[['ZIP', 'YEAR', 'INCOME']]
acs_15 = acs_15.rename(columns={'GEO.id2': 'ZIP', 'HC02_EST_VC02': 'INCOME'})
acs_15 = acs_15[['ZIP', 'YEAR', 'INCOME']]
acs_16 = acs_16.rename(columns={'GEO.id2': 'ZIP', 'HC02_EST_VC02': 'INCOME'})
acs_16 = acs_16[['ZIP', 'YEAR', 'INCOME']]
acs_df = acs_12.append(acs_13, ignore_index=True)
acs_df = acs_df.append(acs_14, ignore_index=True)
acs_df = acs_df.append(acs_15, ignore_index=True)
acs_df = acs_df.append(acs_16, ignore_index=True)
acs_df['ZIP'] = acs_df['ZIP'].astype(str)
# acs_df  


# In[ ]:


# regression 
regression_data['ZIP'] = regression_data['ADDRESS'].str.strip().str[-8:-3]
regression_data['ZIP'] = regression_data['ZIP'].astype(str)
regression_data['DATE'] = regression_data['DATE'].astype(str)
regression_data['YEAR'] = regression_data['DATE'].str.strip().str[:4]
regression_data = regression_data.loc[(regression_data['YEAR'] >= '2012') & (regression_data['YEAR'] <= '2016')]
regression_data


# In[ ]:


# merge all the information to a single table regression_df
# some zip codes are missing in acs data, NEED TO INVESTIGATE !
regression_df = pd.merge(regression_data, acs_df, on=['YEAR','ZIP'])
# use log income in the regression
regression_df['INCOME'] = pd.to_numeric(regression_df['INCOME'], errors='coerce')
regression_df['LOG_INCOME'] = np.log(regression_df['INCOME']) 
regression_df = regression_df[['MORTALITY', 'ASIAN_DUMMY', 'HISPANIC_DUMMY', 'BLACK_DUMMY', 'LOG_INCOME']]
regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
regression_df = regression_df.dropna()
# regression_df


# In[ ]:


X = regression_df[['BLACK_DUMMY', 'ASIAN_DUMMY', 'HISPANIC_DUMMY', 'LOG_INCOME']]
X = add_constant(X)
Y = regression_df['MORTALITY']
model = sm.OLS(Y, X).fit()
model.summary()


# In[ ]:




