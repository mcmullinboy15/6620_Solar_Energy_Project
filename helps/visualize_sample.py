import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("../combined_data.csv", index_col=0, usecols=[1, 2, 3], parse_dates=["date_time"])

for col in data.columns:

    data[col] = preprocessing.scale(data[col])

    plt.plot(data[col], label=col)

plt.legend()
plt.show()
