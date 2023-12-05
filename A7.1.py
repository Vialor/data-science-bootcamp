import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve

glass = pd.read_csv('glass.csv')
glass['household'] = glass.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
glass.sort_values(by = 'Al', inplace=True)
X= np.array(glass.Al).reshape(-1,1)
y = glass.household

logreg = LogisticRegression()
logreg.fit(X, y)

## classification
# pred = logreg.predict(X) # logreg.coef_, logreg.intercept_
# plt.scatter(glass.Al, glass.household)
# plt.plot(glass.Al, pred, color='red')
# plt.xlabel('al')
# plt.ylabel('household')

## probability
# with threshold
# THRESHOLD = 0.5
# pred= logreg.predict_proba(X)[:, 1] >= THRESHOLD
# with probability
pred= logreg.predict_proba(X)[:, 1]

# glass['household_pred_prob'] = pred
# plt.scatter(glass.Al, glass.household)
# plt.plot(glass.Al, glass.household_pred_prob, color='red')
# plt.xlabel('al')
# plt.ylabel('household')
# plt.show()

# print(accuracy_score(y_true=y, y_pred=pred))
# print(precision_score(y_true=y, y_pred=pred))
# print(recall_score(y, pred))

fpr, tpr, threshold = roc_curve(y_true=y, y_score=pred)
print(fpr)
print(tpr)
print(threshold)
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()
