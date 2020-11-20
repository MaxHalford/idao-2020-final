import lightgbm as lgb
import pandas as pd


if __name__ == '__main__':

    model = lgb.Booster(model_file='model.txt')

    test = pd.read_csv(
        'test.csv',
        parse_dates=['epoch'],
        usecols=['id', 'sat_id', 'epoch']
    )

    preds = model.predict(test)

    preds.to_csv('submission.csv', index=False)
