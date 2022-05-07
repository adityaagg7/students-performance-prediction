from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from cgi import print_exception, test
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

df = pd.read_csv('data/StudentsPerformance.csv')
df = df.iloc[:, :-2]
print(df.shape)
print(df.head())
print(df.nunique(0))
print(df.isna().sum())

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['parental level of education'] = le.fit_transform(
    df['parental level of education'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])
df['lunch'] = le.fit_transform(df['lunch'])

print(df.head())

print(df.corr())

x = df.iloc[:, :-1].values
print(x.shape)
y = df.iloc[:, -1].values
print(y.shape)


x_tr, x_t, y_tr, y_t = train_test_split(x, y, test_size=0.2, random_state=69)

reg = LinearRegression()
reg.fit(x_tr, y_tr)
y_pr = reg.predict(x_t)

pt.scatter(y_t, y_t - y_pr)
pt.xlabel('True Value')
pt.ylabel('Error')
pt.show()


r2 = r2_score(y_t, y_pr)
print("R2 for maths score:", r2)
