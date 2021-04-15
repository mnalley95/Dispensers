# %%
from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.common import load_datasets, ListDataset
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


from gluonts.dataset.common import (
    BasicFeatureInfo,
    CategoricalFeatureInfo,
    ListDataset,
    MetaData,
    TrainDatasets)

from gluonts.dataset.field_names import FieldName
from tqdm.autonotebook import tqdm

from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

from Processing import (
    processed_df,
    processed_df_fill)
from typing import Any, Callable, Dict, Sequence

# %%
single_prediction_length = 6
submission_prediction_length = single_prediction_length * 2
submission = True
m5_input_path = './'

if submission:
    prediction_length = submission_prediction_length
else:
    prediction_length = single_prediction_length

# %%

cal_features = calendar.drop(
    ['date', 'wm_yr_wk', 'weekday', 'wday', 'month',
        'year', 'event_name_1', 'event_name_2', 'd'],
    axis=1
)
cal_features['event_type_1'] = cal_features['event_type_1'].apply(
    lambda x: 0 if str(x) == "nan" else 1)
cal_features['event_type_2'] = cal_features['event_type_2'].apply(
    lambda x: 0 if str(x) == "nan" else 1)

test_cal_features = cal_features.values.T
if submission:
    train_cal_features = test_cal_features[:, :-submission_prediction_length]
else:
    train_cal_features = test_cal_features[:, :-
                                           submission_prediction_length-single_prediction_length]
    test_cal_features = test_cal_features[:, :-submission_prediction_length]

test_cal_features_list = [test_cal_features] * len(sales_train_validation)
train_cal_features_list = [train_cal_features] * len(sales_train_validation)

# %%
item_ids = processed_df_fill["sku"].astype('category').cat.codes.values
item_ids_un, item_ids_counts = np.unique(item_ids, return_counts=True)

cust_ids = processed_df_fill["Custname"].astype('category').cat.codes.values
cust_ids_un, cust_ids_counts = np.unique(cust_ids, return_counts=True)

cat_ids = processed_df_fill["Label"].astype('category').cat.codes.values
cat_ids_un, cat_ids_counts = np.unique(cat_ids, return_counts=True)

stat_cat_list = [item_ids, cust_ids, cat_ids]

stat_cat = np.concatenate(stat_cat_list)
stat_cat = stat_cat.reshape(len(stat_cat_list), len(item_ids)).T

stat_cat_cardinalities = [len(item_ids_un), len(cust_ids_un), len(cat_ids_un)]


# %%

train_df = processed_df_fill.pivot_table(
    'y', index=['Custname', 'sku', 'Label'], columns='x')
train_target_values = train_df.values

if submission == True:
    test_target_values = [np.append(ts, np.ones(
        submission_prediction_length) * np.nan) for ts in train_df.values]
else:
    test_target_values = train_target_values.copy()
    train_target_values = [ts[:-single_prediction_length]
                           for ts in train_df.values]

data_iter = []

# Build the payload
for item_id, dfg in processed_df_fill.groupby(['Custname', 'sku', 'Label'], as_index=False):
    data_iter.append(
        pd.Timestamp(dfg.iloc[0]["x"], freq='W')
    )

m5_dates = data_iter

#m5_dates = [pd.Timestamp("2017-01-01", freq='1W') for _ in range(len(train_target_values))]

train_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        # FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, start, fsc) in zip(train_target_values,
                                    m5_dates,
                                    # train_cal_features_list,
                                    stat_cat)
], freq="W")

test_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        # FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, start, fsc) in zip(test_target_values,
                                    m5_dates,
                                    # test_cal_features_list,
                                    stat_cat)
], freq="W")
# %%
next(iter(train_ds))
# %%

estimator = DeepAREstimator(
    prediction_length=prediction_length,
    freq="W",
    # use_feat_dynamic_real=True,
    use_feat_static_cat=True,
    cardinality=stat_cat_cardinalities,
    trainer=Trainer(
        learning_rate=1e-3,
        epochs=3,
        num_batches_per_epoch=50,
        batch_size=32
    )
)

predictor = estimator.train(train_ds)
# %%


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

print("Obtaining time series conditioning values ...")
#tss = list(tqdm(ts_it, total=len(test_ds)))
print("Obtaining time series predictions ...")
#forecasts = list(tqdm(forecast_it, total=len(test_ds)))
# %%
test_entry = next(iter(forecast_it))


def sample_df(forecast):
    samples = forecast.samples
    ns, w = samples.shape
    dates = pd.date_range(forecast.start_date, freq=forecast.freq, periods=w)
    return pd.DataFrame(samples.T, index=dates)


# %%
# https://stackoverflow.com/questions/61416951/export-multiple-gluonts-forecasts-to-pandas-dataframe
parts = [sample_df(entry).assign(entry=i)
         for i, entry in enumerate(forecast_it)]

long_form = pd.concat(parts).reset_index().melt(['index', 'entry'])
long_form.rename(columns={
    'index': 'ts',
    'variable': 'sample',
    'value': 'forecast',
})

long_form = long_form.groupby(['index', 'entry']).mean(
).reset_index().sort_values(['entry', 'index'])
