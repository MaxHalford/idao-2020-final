import glob

import joblib
import lightgbm as lgb
import pandas as pd

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


class CountEncoder:

    def __init__(self, df, on):
        self.on = on
        self.counts = df.groupby(on).size().rename(str(self))

    def __str__(self):
        return f'{self.on}_size'

    def transform(self, df):
        return df.join(self.counts, on=self.on)[self.counts.name]


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
    'clnt_income_month_avg_net_amt': 'uint',
    'clnt_expense_month_avg_amt': 'uint',
    'app_addr_region_reg': 'category',
    'app_addr_region_fact': 'category',
    'app_addr_region_sale': 'category',
    'prt_name': 'category',
    'feature_0': 'category',
    'clnt_birth_year': 'uint',
    'addr_region_fact_encoding1': 'float32',
    'addr_region_fact_encoding2': 'float32',
    'addr_region_reg_encoding1': 'float32',
    'addr_region_reg_encoding2': 'float32',
    'app_addr_region_reg_encoding1': 'float32',
    'app_addr_region_reg_encoding2': 'float32',
    'app_addr_region_fact_encoding1': 'float32',
    'app_addr_region_fact_encoding2': 'float32',
    'app_addr_region_sale_encoding1': 'float32',
    'app_addr_region_sale_encoding2': 'float32',
    'loans_main_borrower': 'float32',
    'loans_active': 'float32',
    'max_overdue_status': 'category',
    'ttl_inquiries': 'uint',
    'ttl_auto_loan': 'bool',
    'ttl_mortgage': 'bool',
    'ttl_credit_card': 'bool',
    'ttl_consumer': 'bool',
    'worst_status_ever': 'category',
    'feature_0': 'category',
    'feature_1': 'uint',
    'feature_2': 'uint',
    'feature_3': 'uint',
    'feature_4': 'uint',
    'feature_5': 'uint',
    'feature_6': 'uint',
    'feature_7': 'uint',
    'fl_coborrower': 'bool',
    'fl_active_coborrower': 'bool',
    'pay_load': 'float32',
    'makro_region': 'category',
    'fo': 'category',
    'region': 'category',
    'feature_30': 'category'
}

fill_missing = [
    ('inquiry_recent_period', 0, 'uint'),
    ('inquiry_1_week', 0, 'uint'),
    ('clnt_experience_cur_mnth', 0, 'uint'),
    ('clnt_experience_total_mnth', 0, 'uint'),
    ('last_loan_date', 6200, 'uint'),
    ('first_loan_date', 6200, 'uint'),
    ('inquiry_recent_period', 0, 'uint'),
    ('inquiry_3_month', 0, 'uint'),
    ('inquiry_6_month', 0, 'uint'),
    ('inquiry_9_month', 0, 'uint'),
    ('inquiry_12_month', 0, 'uint'),
    ('inquiry_1_week', 0, 'uint'),
    ('inquiry_1_month', 0, 'uint'),
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

df['n_missing'] = df.isnull().sum(axis='columns')

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
prediction['target'] = model.predict(test)
quantile_inf = prediction['target'].quantile(0.05)
quantile_sup = prediction['target'].quantile(0.95)
prediction['target'] = prediction['target'].apply(
    lambda x: 0.00000003 if x < quantile_sup and x > quantile_inf else x)
prediction.to_csv('prediction.csv', index=False)
