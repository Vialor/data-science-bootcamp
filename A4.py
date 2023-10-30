import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the Boston housing data.
housing_csv = 'data/boston_housing_data.csv'
housing = pd.read_csv(housing_csv)

# Read in the drinks data.
drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
url = 'data/drinks.csv'
drinks = pd.read_csv(url, header=0, names=drink_cols, na_filter=False)
plt.style.use('fivethirtyeight')

# 1
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
housing.ZN.value_counts().sort_index().plot(color="green")
housing.INDUS.value_counts().sort_index().plot(color="blue", linestyle='dashed')
plt.show()

# 2
columns = ['col1', 'col2', 'col3', 'col4']
index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(np.random.randn(10, 4), columns=columns, index=index)
ax = df[['col1', 'col2']].plot(kind='barh', figsize=(9, 5))
ax.set_title('Some Random HUGE Title', fontsize=40,y=1)
# lower left
# ax.legend(loc=3)
# horizontal
# use this .plot(kind='barh') instead of 'bar'
# upper right
# ax.legend(loc=1)
plt.show()

# 3
housing.MEDV.plot(kind='hist', bins=20)
plt.show()

# 4 Create a scatter plot of two heatmap entries that appear to have a very positive correlation.
housing_correlations = housing.corr()
sns.heatmap(housing_correlations)

housing[['CRIM', 'TAX']].plot(kind='scatter', x='CRIM', y='TAX')
plt.show()

# 5 Now, create a scatter plot of two heatmap entries that appear to have negative correlation.
housing[['AGE', 'DIS']].plot(kind='scatter', x='AGE', y='DIS')
plt.show()