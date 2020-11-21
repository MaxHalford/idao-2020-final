import lightgbm as lgb
import pandas as pd

dtypes = {
    'delivery_type': 'category',
    'addr_region_reg': 'category',
    'addr_region_fact': 'category',
    'channel_name': 'category',
    'channel_name_2': 'category',
    'sas_limit_after_003_amt': 'uint8',
    'sas_limit_last_amt': 'uint8'
}
cols_to_use = list(dtypes.keys()) + ['card_id']

test = pd.read_csv(
    'test.csv',
    #'../data/train.csv',
    index_col='card_id',
    dtype=dtypes,
    usecols=cols_to_use
)

model = lgb.Booster(model_file='model.lgb')

prediction = test.index.to_frame()
prediction['target'] = model.predict(test)
prediction.to_csv('prediction.csv', index=False)
