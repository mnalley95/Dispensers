#%%
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gluonts.dataset.common import (
    BasicFeatureInfo, 
    CategoricalFeatureInfo,
    ListDataset,
    MetaData,
    TrainDatasets)

from gluonts.dataset.field_names import FieldName

from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

from Processing import (
    processed_df,
    processed_df_fill)
from typing import Any, Callable, Dict, Sequence


#%%
def encode_cat(cats):
    return {c: i for i, c in enumerate(cats)}


def df2gluonts(
    df,
    cat_idx,
    fcast_len: int,
    freq: str = "D",
    ts_id: Sequence[str] = ["cat", "cc"],
    static_cat: Sequence[str] = ["cat", "cc"],
    item_id_fn: Callable = None,
) -> None:
    """Convert a dataframe of multiple timeseries to json lines.

    This function supports gluonts static features, but not the dynamic features.

    Args:
        df (pd.DataFrame): Dataframe of multiple timeseries, where target variable must be called column `y`.
        cat_idx (Dict[str, Dict[str, int]]): Mapper for static categories.
        fcast_len (int, optional): Forecast horizon. Defaults to 12.
        freq (str, optional): Frequency of timeseries. Defaults to 'W'.
        ts_id (Sequence[str], optional): Identifier columns in the dataframe. Defaults to ['cat', 'cc'].
        static_cat (Sequence[str], optional): Columns that denotes static category features of each timeseries.
            Defaults to ['cat', 'cc'].
        item_id_fn ([type], optional): Function to format `item_id`. Defaults to None.
    """
    data_iter = []

    # Build the payload
    for item_id, dfg in df.groupby(ts_id, as_index=False):
        if len(ts_id) < 2:
            item_id = [item_id]

        if fcast_len > 0:
            # Train split exclude the last fcast_len timestamps
            ts_len = len(dfg) - fcast_len
            target = dfg["y"][:-fcast_len]
        else:
            # Test split include all timeseries. During backtesting,
            # gluonts will treat the fcast_len as groundtruth.
            target = dfg["y"]

        feat_static_cat = []
        for col in static_cat:
            # Construct all static category features of current timeseries.
            assert dfg[col].nunique() == 1
            cat_value = dfg[col].iloc[0]
            # Encode sku to zero-based number for feat_static_cat.
            feat_static_cat.append(cat_idx[col][cat_value])

        if item_id_fn is None:
            # NOTE: our sm-glounts entrypoint will interpret '|' as '\n'
            # in the plot title.
            item_id = "|".join(item_id)
        else:
            item_id = item_id_fn(*item_id)

        data_iter.append(
            {"start": dfg.iloc[0]["x"], "target": target, "feat_static_cat": feat_static_cat, "item_id": item_id}
        )

    # Finally we call gluonts API to convert data_iter with frequency of
    # the observation in the time series
    data = ListDataset(data_iter, freq=freq)
    return data


#%%
freq, fcast_length = 'W', 12

cat_inverted_idx = {'sku': encode_cat(processed_df_fill['sku'].unique()), 'Label': encode_cat(processed_df_fill['Label'].unique())}




# Drop the final fcast_length from train data.
train_data= df2gluonts(processed_df_fill,
                       cat_inverted_idx,
                       fcast_len=fcast_length,
                       freq=freq,
                       ts_id=['sku', 'Label'],
                       static_cat=['sku', 'Label']
)

# Test data include fcast_length which are ground truths.
test_data = df2gluonts(processed_df_fill,
                       cat_inverted_idx,
                       fcast_len=0,
                       freq=freq,
                       ts_id=['sku', 'Label'],
                       static_cat=['sku', 'Label']
)




gluonts_datasets = TrainDatasets(
    metadata=MetaData(
                freq=freq,
                target={'name': 'quantity'},
                feat_static_cat=[
                    CategoricalFeatureInfo(name=k, cardinality=len(v)+1)   # Add 'unknown'.
                    for k,v in cat_inverted_idx.items()
                ],
                prediction_length = fcast_length
    ),
    train=train_data,
    test=test_data
)   

#%%
epochs = 10
fcast_length = 12

metric=[
    {"Name": "train:loss", "Regex": r"Epoch\[\d+\] Evaluation metric 'epoch_loss'=(\S+)"},
    {"Name": "train:learning_rate", "Regex": r"Epoch\[\d+\] Learning rate is (\S+)"},
    {"Name": "test:abs_error", "Regex": r"gluonts\[metric-abs_error\]: (\S+)"},
    {"Name": "test:rmse", "Regex": r"gluonts\[metric-RMSE\]: (\S+)"},
    {"Name": "test:mape", "Regex": r"gluonts\[metric-MAPE\]: (\S+)"},
    {"Name": "test:smape", "Regex": r"gluonts\[metric-sMAPE\]: (\S+)"},
    {"Name": "test:wmape", "Regex": r"gluonts\[metric-wMAPE\]: (\S+)"},
]


from gluonts.model import deepar

estimator = deepar.DeepAREstimator(
    prediction_length=gluonts_datasets.metadata.prediction_length,
    freq= gluonts_datasets.metadata.freq,
    cardinality= [gluonts_datasets.metadata.feat_static_cat[0].cardinality],
    use_feat_static_cat=True,
    #use_feat_dynamic_real=True,
    #use_feat_dynamic_cat = True,
    #use_feat_static_real=True,
    #time_features= time_features,
    trainer= Trainer(epochs = epochs)
)



predictor = estimator.train(training_data=gluonts_datasets.train, validation_data=gluonts_datasets.test)


#%%

forecast_it, ts_it = make_evaluation_predictions(
    dataset=gluonts_datasets.test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=500,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)
tss = list(ts_it)


#evaluations 
#%%
from gluonts.evaluation import Evaluator
import json

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(gluonts_datasets.test))

print(json.dumps(agg_metrics, indent=4))

#%%

item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
plt.grid(which="both")
plt.show()


# %%
def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.yaxis.set_major_formatter(mlp.ticker.StrMethodFormatter('{x:,.0f}'))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.title(forecast_entry.item_id)
    plt.legend(legend, loc="upper left")
    plt.show()
# %%
ts_entry = next(iter(tss))
forecast_entry = next(iter(forecasts))
# %%
plot_prob_forecasts(ts_entry,forecast_entry)
# %%
for i in range(0,len(forecasts), 5):
    '''
    plot forecasts
    '''
    plot_prob_forecasts(tss[i],forecasts[i])
# %%

forecasts = list(predictor.predict(dataset=gluonts_datasets.test))

# %%
for i in range(0,len(forecasts)):
    '''
    plot forecasts
    '''
    if 'Walmart' in forecasts[i].item_id:
        plot_prob_forecasts(tss[i],forecasts[i])
    else:
        pass
# %%
