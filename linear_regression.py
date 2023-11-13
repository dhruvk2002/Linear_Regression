import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('dark_background')
df = quandl.get('WIKI/GOOGL')
df=df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_Change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0
df= df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
forecast_col='Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out= int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)


X= np.array(df.drop('label', axis=1))
X= preprocessing.scale(X)
X_lately=X[-forecast_out:]
X=X[:-forecast_out]

df.dropna(inplace=True)
y= np.array(df['label'])

# Saving the model

X_train, X_test, y_train, y_test= model_selection.train_test_split(X, y, test_size=0.2)
# classifier= LinearRegression(n_jobs=-1)
# classifier.fit(X_train, y_train)
# with open('linear_regression.pickel', 'wb') as f:
#     pickle.dump(classifier, f)

# Loading the model

model_path=open('Linear_Regression/linear_regression.pickel', 'rb')
classifier=pickle.load(model_path)
# confidence= classifier.score(X_test, y_test, sample_weight=None)

# print(forecast_out)
# print(confidence)

prediction_set=classifier.predict(X_lately)

# print(prediction_set, confidence, forecast_out)

df['Forecast']=np.nan
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
next_unix=last_date+pd.DateOffset(days=1)
for i in prediction_set:
    next_date= datetime.datetime.timestamp(next_unix)
    next_unix+=pd.DateOffset(days=1)
    df.loc[next_date]= [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

