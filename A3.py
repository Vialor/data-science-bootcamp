import numpy as np
import pandas as pd

# Numpy
# 1
a = np.arange(5)
b = np.arange(3, 8)
abvstack = np.vstack((a, b))
abhstack = np.hstack((a, b))
print(abvstack)
print(abhstack)
# 2
print(np.intersect1d(a, b))
# 3: less than 3
print(a[a < 3])
# 4
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
mask = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
print(iris_2d[mask])

# Pandas
# 1
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
print(df.iloc[::20, :3])
# 2
df_price = df[['Min.Price', 'Max.Price']]
df_price = df_price.apply(lambda x: x.fillna(x.mean()))
print(df_price)
# 3
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
mask = df.apply(sum, axis=1) > 100
print(df[mask])
