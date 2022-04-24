from cgi import print_exception
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

x = df[:, [0]]
# y = df['math score', 'reading score', 'writing score']
print(x.head())
print(y.head())