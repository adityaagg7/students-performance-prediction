from cgi import print_exception, test
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

df = pd.read_csv('data/StudentsPerformance.csv')

print(df.shape)
print(df.head())
print(df.nunique(0))
print(df.isna().sum())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['parental level of education'] = le.fit_transform(
    df['parental level of education'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])
df['lunch'] = le.fit_transform(df['lunch'])

print(df.head())

print(df.corr())

x = df.iloc[:, :-3].values
print(x)
y = df.iloc[:, -3:]
print(y)

from sklearn.model_selection import train_test_split

x_tr, x_t, y_tr, y_t = train_test_split(x, y, test_size=0.2, random_state=69)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_tr, y_tr)
y_pr = reg.predict(x_t)

from sklearn.metrics import r2_score

r2 = r2_score(y_t, y_pr)
print(r2)
