#%%
#packages
import pandas as pd
import numpy as np
import os
from gluonts_nb_utils import fill_dt_all
#%%
#import data
raw_data = pd.read_csv("../Data/DispenserWeek.csv")

#%%
#parameters
start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2021-03-01')

#%%
#clean and format
raw_data["Glpostdt"] = pd.to_datetime(raw_data['Glpostdt']).dt.tz_convert(None)
raw_data = raw_data[(raw_data["Glpostdt"] > start_date) & (raw_data["Glpostdt"] < end_date)]

#%%
#round dates
raw_data['Glpostdt'] = raw_data['Glpostdt'].dt.to_period('W').dt.to_timestamp()
#%%
#sum quantity by group
grouped_df=raw_data.groupby(["Custname","Itemnmbr","Glpostdt"])["Quantity"].sum()


#%%
#final formatting
processed_df= grouped_df.reset_index()
processed_df['Label'] = np.where(processed_df['Custname'].str.contains('com'), 'COM', np.where(processed_df['Custname'].str.contains('di') | processed_df['Custname'].str.contains('Direct Import'), 'DI', 'Domestic'))
processed_df['sku'] = processed_df['Itemnmbr'].astype('str')
processed_df = processed_df.drop(['Itemnmbr'], axis = 1).rename({'Glpostdt': 'x', 'Quantity': 'y'}, axis = 1)

#clean inactive products
date_check = np.datetime64('2021-01-01')
processed_df = processed_df.groupby(['sku', 'Custname', 'Label']).filter(lambda d: max(d['x']) > date_check).reset_index(drop = True)

#get groups with more than 6 rows
group_size = processed_df.groupby(['Custname','Label','sku']).size()
group_size = pd.DataFrame(group_size).rename({0:'size'}, axis=1)
group_size = group_size[group_size['size'] > 12]

#filter based upon index of group_size
index1 = processed_df.set_index(['Custname', 'Label', 'sku']).index
index2 = group_size.index

processed_df = processed_df[index1.isin(index2)]
# %%
freq = 'W'

processed_df_fill = fill_dt_all(processed_df, ts_id=['sku', 'Label', 'Custname'], dates = ('min', end_date, 'D'), freq = freq)