#%%
#packages
import pandas as pd
import numpy as np
import os
from gluonts_nb_utils import fill_dt_all
#%%
#import canada data
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
processed_df['sku'] = processed_df['Custname'] + '_' + processed_df['Itemnmbr'].astype('str')
processed_df = processed_df.drop(['Custname', 'Itemnmbr'], axis = 1).rename({'Glpostdt': 'x', 'Quantity': 'y'}, axis = 1)

#clean inactive products
date_check = np.datetime64('2021-01-01')
processed_df = processed_df.groupby(['sku']).filter(lambda d: max(d['x']) > date_check).reset_index(drop = True)
# %%
freq = 'W'

processed_df_fill = fill_dt_all(processed_df, ts_id=['sku'], dates = ('min', end_date, 'D'), freq = freq)

# %%
