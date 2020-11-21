import glob

import joblib
import lightgbm as lgb
import pandas as pd
from scipy import special

class MeanEncoder:

    def __init__(self, df, on, by, prior_count=100):
        self.on = on
        self.by = by
        counts = df.groupby(by)[on].agg(['mean', 'count'])
        avg = df[on].mean()
        self.means = (
            counts
            .eval('(mean * count + @avg * @prior_count) / (count + @prior_count)')
            .rename(str(self))
        )

    def __str__(self):
        return f'avg_{self.on}_by_{self.by}'

    def transform(self, df):
        return df.join(self.means, on=self.by)[self.means.name]

class NMissing:

    def __str__(self):
        return 'n_missing'

    def transform(self, df):
        return df.isnull().sum(axis='columns').rename(str(self))

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

extractors = [
    joblib.load(path)
    for path in glob.glob('*.pkl')
]

features = []

for ex in extractors:
    features.append(ex.transform(df))

test = pd.concat([df] + features, axis='columns')

model = lgb.Booster(model_file='model.lgb')

prediction = df.index.to_frame()
prediction['target'] = special.expit(model.predict(test))
prediction.to_csv('prediction.csv', index=False)
