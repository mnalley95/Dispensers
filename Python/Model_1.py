#%%
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

#%%
single_prediction_length = 6   
submission_prediction_length = single_prediction_length * 2
submission=True
m5_input_path = './'

if submission:
    prediction_length = submission_prediction_length
else:
    prediction_length = single_prediction_length

#%%
item_ids = processed_df_fill["sku"].astype('category').cat.codes.values
item_ids_un , item_ids_counts = np.unique(item_ids, return_counts=True)

cust_ids = processed_df_fill["Custname"].astype('category').cat.codes.values
cust_ids_un , cust_ids_counts = np.unique(cust_ids, return_counts=True)

cat_ids = processed_df_fill["Label"].astype('category').cat.codes.values
cat_ids_un , cat_ids_counts = np.unique(cat_ids, return_counts=True)

stat_cat_list = [item_ids, cust_ids, cat_ids]

stat_cat = np.concatenate(stat_cat_list)
stat_cat = stat_cat.reshape(len(stat_cat_list), len(item_ids)).T

stat_cat_cardinalities = [len(item_ids_un), len(cust_ids_un), len(cat_ids_un)]


#%%
from gluonts.dataset.common import load_datasets, ListDataset
from gluonts.dataset.field_names import FieldName

train_df = processed_df_fill.pivot_table('y', index = ['Custname','sku','Label'], columns = 'x')
train_target_values = train_df.values

if submission == True:
    test_target_values = [np.append(ts, np.ones(submission_prediction_length) * np.nan) for ts in train_df.values]
else:
    test_target_values = train_target_values.copy()
    train_target_values = [ts[:-single_prediction_length] for ts in train_df.values]

data_iter = []

# Build the payload
for item_id, dfg in processed_df_fill.groupby(['Custname', 'sku', 'Label'], as_index=False):
    data_iter.append(
        pd.Timestamp(dfg.iloc[0]["x"], freq = 'W')
    )

m5_dates = data_iter

#m5_dates = [pd.Timestamp("2017-01-01", freq='1W') for _ in range(len(train_target_values))]

train_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        #FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, start, fsc) in zip(train_target_values,
                                         m5_dates,
                                         #train_cal_features_list,
                                         stat_cat)
], freq="W")

test_ds = ListDataset([
    {
        FieldName.TARGET: target,
        FieldName.START: start,
        #FieldName.FEAT_DYNAMIC_REAL: fdr,
        FieldName.FEAT_STATIC_CAT: fsc
    }
    for (target, start, fsc) in zip(test_target_values,
                                         m5_dates,  
                                         #test_cal_features_list,
                                         stat_cat)
], freq="W")
# %%
next(iter(train_ds))
# %%
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

estimator = DeepAREstimator(
    prediction_length=prediction_length,
    freq="W",
    #use_feat_dynamic_real=True,
    use_feat_static_cat=True,
    cardinality=stat_cat_cardinalities,
    trainer=Trainer(
        learning_rate=1e-3,
        epochs=10,
        num_batches_per_epoch=50,
        batch_size=32
    )
)

predictor = estimator.train(train_ds)
# %%

from gluonts.evaluation.backtest import make_evaluation_predictions

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

print("Obtaining time series conditioning values ...")
tss = list(tqdm(ts_it, total=len(test_ds)))
print("Obtaining time series predictions ...")
forecasts = list(tqdm(forecast_it, total=len(test_ds)))
# %%
if submission == True:
    forecasts_acc = np.zeros((len(forecasts), prediction_length))
    for i in range(len(forecasts)):
        forecasts_acc[i] = np.mean(forecasts[i].samples, axis=0)
# %%
if submission == True:
    forecasts_acc_sub = np.zeros((len(forecasts)*2, single_prediction_length))
    forecasts_acc_sub[:len(forecasts)] = forecasts_acc[:,:single_prediction_length]
    forecasts_acc_sub[len(forecasts):] = forecasts_acc[:,single_prediction_length:]

#%%
if submission == True:
    np.all(np.equal(forecasts_acc[0], np.append(forecasts_acc_sub[0], forecasts_acc_sub[161])))

#%%

if submission == True:
    import time

    #sample_submission = pd.read_csv('sample_submission.csv')
    #sample_submission.iloc[:,1:] = forecasts_acc_sub
    sample_submission = forecasts_acc_sub
    submission_id = 'submission_{}.csv'.format(int(time.time()))

    sample_submission.to_csv(submission_id, index=False)
# %%
plot_log_path = "./plots/"
directory = os.path.dirname(plot_log_path)
if not os.path.exists(directory):
    os.makedirs(directory)
    
def plot_prob_forecasts(ts_entry, forecast_entry, path, sample_id, inline=True):
    plot_length = 150
    prediction_intervals = (50, 67, 95, 99)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    _, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    ax.axvline(ts_entry.index[-prediction_length], color='r')
    plt.legend(legend, loc="upper left")
    if inline:
        plt.show()
        plt.clf()
    else:
        plt.savefig('{}forecast_{}.pdf'.format(path, sample_id))
        plt.close()

print("Plotting time series predictions ...")
for i in tqdm(range(5)):
    ts_entry = tss[i]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry, plot_log_path, i)
# %%
