import lightgbm as lgb
import pandas as pd

dtypes = {
    'delivery_type': 'category',
    'addr_region_reg': 'category',
    'addr_region_fact': 'category',
    'channel_name': 'category',
    'channel_name_2': 'category',
    'sas_limit_after_003_amt': 'uint8',
    'sas_limit_last_amt': 'uint8',
    'channel_name_modified_2018': 'category',
    'clnt_education_name': 'category',
    'clnt_marital_status_name': 'category',
    'clnt_employment_type_name': 'category',
    'clnt_speciality_sphere_name': 'category',
    'clnt_sex_name': 'category',
    'prt_name': 'category',
    'feature_0': 'category',
}

fill_missing = [
    ('inquiry_recent_period', 0, 'uint'),
    ('inquiry_1_week', 0, 'uint')
]

cols_to_use = (
    list(dtypes.keys()) +
    list(col for col, *_ in fill_missing) +
    ['card_id']
)

df = pd.read_csv(
    'test.csv',
    index_col='card_id',
    dtype=dtypes,
    usecols=cols_to_use
)

for col, fill, dtype in fill_missing:
    df[col] = df[col].fillna(fill).astype(dtype)

model = lgb.Booster(model_file='model.lgb')

prediction = df.index.to_frame()
prediction['target'] = model.predict(test)
prediction.to_csv('prediction.csv', index=False)
