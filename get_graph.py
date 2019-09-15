import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from countvectorizer import *
import seaborn as sns


df = pd.read_csv('bbc-text.csv')

print(df.head())
print(df.describe())

x = df['text']
y = df['category']

sns.set_palette("bright")


sns.countplot(y)
plt.show()