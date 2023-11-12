import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 101)

data = pd.read_csv("./data/train.csv")
data = data.drop(columns=['id', 'timestamp','country'])

data.loc[data['hours_per_week'].isna(), 'hours_per_week'] = data['hours_per_week'].median()
data.loc[data['telecommute_days_per_week'].isna(), 'telecommute_days_per_week'] = data['telecommute_days_per_week'].median()
data = data.dropna()

# encode categorical features
data_train = data.copy()
cat_cols = [c for c in data_train.columns if data_train[c].dtype == 'object' 
            and c not in ['is_manager', 'certifications']]
cat_data = data_train[cat_cols]
binary_cols = ['is_manager', 'certifications']
for c in binary_cols:
    data_train[c] = data_train[c].replace(to_replace=['Yes'], value=1)
    data_train[c] = data_train[c].replace(to_replace=['No'], value=0)
final_data = pd.get_dummies(data_train, columns=cat_cols, drop_first= True)

# train test split
y = final_data['salary']
X = final_data.drop(columns=['salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# normalize data
num_cols = ['job_years','hours_per_week','telecommute_days_per_week']
scaler = StandardScaler()
scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# linear regression training
reg = LinearRegression()
reg.fit(X_train, y_train)

# prediction
y_pred = reg.predict(X_test) # reg.coef_, reg.intercept_
print(mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred)**0.5)

# use Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

# plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test), label='true')
# plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred), label = 'pred')
# plt.legend()

# use Lasso
lasso = Lasso(alpha=1)
lasso.fit(X_train,y_train)
y_pred = lasso.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)
