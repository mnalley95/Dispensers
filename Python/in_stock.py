#%%
import pandas as pd
import pyodbc
import plotly.express as px

cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=192.168.66.59;"
                      "Database=Data Warehouse;"
                      "Trusted_Connection=yes;")

#%%
df = pd.read_sql_query("SELECT [Retailer], [PrimoItem], [ItemDesc], [ItemType], [WeekEnding], SUM(InStock)/COUNT(InStock) AS 'In Stock Percentage' FROM [Data Warehouse].[dbo].[WSUImport] WHERE [WeekEnding] >= '2020-01-01' AND [Retailer] = 'WALM' GROUP BY [Retailer], [PrimoItem], [ItemDesc], [ItemType], [WeekEnding] ORDER BY [WeekEnding]", cnxn)


#%%
df.ItemDesc = df.ItemDesc.replace('PRIMO MANUAL PUMP WC', 'PRIMO WATER PUMP')
df.ItemDesc = df.ItemDesc.replace('PRIMO TABLETOP DISP', 'PRIMO BLKTBTOP CROCK')

#%%


df.WeekEnding.max()

top_items = ['601325',
'900179',
'601346',
'601148',
'601354',
'601244',
'900127',
'601242',
'601258',
'900130',
'601243',
'601323',
'601305']


df_top = df[df['PrimoItem'].isin(top_items)]

#%%

px.line(df_top, x = 'WeekEnding', y = 'In Stock Percentage', color = 'ItemDesc')

recent = df_top[df_top['WeekEnding'] >= pd.to_datetime('2021-01-01')]

fig = px.line(recent, x = 'WeekEnding', y = 'In Stock Percentage',  facet_col_wrap= 3, facet_col= 'ItemDesc')
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(title = '')
fig.update_xaxes(title = '')
fig.update_yaxes(tickformat = '%')
fig.update_layout(title='In-Stock by Item at Walmart')
fig.show()

# %%

fig.write_image('plots/instock.png', width = 1000, height = 500, scale = 1.3)
# %%
