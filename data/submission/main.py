import numpy as np
import pandas as pd

test = pd.read_csv("test.csv")

# just sending random values as the answer
prediction = test[["card_id"]].copy(deep=True)
np.random.seed(42)
prediction["target"] = np.random.rand(test.shape[0])
prediction.to_csv("prediction.csv", index=False)
