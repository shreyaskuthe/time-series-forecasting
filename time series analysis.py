#consists of two things #analysis #forecasting
##work on historical data containing time
#data should be singal dimentional - no independent variable is considered only y variable is considered
##cronological order
##should have equally spaced time interval data
#exponential smoothing, average,
#exponential smoothing#simple exponential smoothing#doubble oreder exp smoothing (holts linear trend model)
#holt winter model (seaspon factor)
#arima family model
#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_excel(r'E:\DATA SCIENCE\imarticus\python\datasets\Sample - Superstore.xls')
pd.set_option('display.max_columns',None)
df.head()
#%%
print(df.shape)
#%% to check discreart variables in category column
print(df.Category.value_counts())
#%%
furniture=df.loc[df['Category']=='Furniture']
furniture.shape
#%%
print(furniture['Order Date'].min(), furniture['Order Date'].max())
#%%
furniture=furniture[["Order Date","Sales"]]
furniture=furniture.sort_values('Order Date')
furniture.isnull().sum()
#%%
furniture.head(10)
#%%
furniture=furniture.groupby('Order Date')['Sales'].sum().reset_index()
furniture.shape
#%%
furniture.head(10)
#%%
furniture=furniture.set_index('Order Date')
furniture.head()
#%%
y=furniture['Sales'].resample('MS').mean()
#%%
print(y.shape)
#%%
y.plot(figsize=(10,6))
plt.show
#%%
train=y.loc[:'2016-12-01']
train.tail()


test=y.loc['2017-01-01':]
test.tail()
#%%
#plotting data
train.plot(figsize=(10,6),title='Average Sales',fontsize=14)
test.plot(figsize=(10,6),title='Average Sales',fontsize=14)
plt.show()
#%% value close to zero older values are less important  compared to recent value&value close to 1 are equally important
from statsmodels.tsa.api import SimpleExpSmoothing
Exp_Smooth = test.copy()
fit1=SimpleExpSmoothing(train).fit(smoothing_level=0.1)
Exp_Smooth['SES']= fit1.forecast(len(test))


train.plot(figsize=(10,6), title='Average Sales',fontsize=14)
test.plot(figsize=(10,6), title='Average Sales',fontsize=14)
Exp_Smooth['SES'].plot(figsize=(10,6), title='Average Sales',fontsize=14)
plt.show
#%%
from sklearn.metrics import mean_squared_error
from math import sqrt
rms=sqrt(mean_squared_error(test,Exp_Smooth.SES))
print(rms)
#%%
#%%
#%%
#%%
#%% seasonal decomposional plot
import statsmodels.api as sm 
decomposition=sm.tsa.seasonal_decompose(y)
fig=decomposition.plot()
plt.show()
#%%
from statsmodels.tsa.api import Holt
Exp_Smooth = test.copy()
fit1=Holt(train).fit(smoothing_level=0.05,smoothing_slope=0.7)
Exp_Smooth['Holt_linear']= fit1.forecast(len(test))


train.plot(figsize=(10,6), title='Average Sales',fontsize=14)
test.plot(figsize=(10,6), title='Average Sales',fontsize=14)
Exp_Smooth['Holt_linear'].plot(figsize=(10,6), title='Average Sales',fontsize=14)
plt.show
#%%
from sklearn.metrics import mean_squared_error
from math import sqrt
rms=sqrt(mean_squared_error(test,Exp_Smooth.Holt_linear))
print(rms)
#%% values exponentially increasing we build multiplicative moddel
from statsmodels.tsa.api import ExponentialSmoothing
Hot_Winter_df = test.copy()
fit1=ExponentialSmoothing(train,seasonal_periods=12,
                 trend='add',seasonal='add').fit()
Exp_Smooth['Holt_winter']= fit1.forecast(len(test))


train.plot(figsize=(10,6), title='Average Sales',fontsize=14)
test.plot(figsize=(10,6), title='Average Sales',fontsize=14)
Exp_Smooth['Holt_winter'].plot(figsize=(10,6), title='Average Sales',fontsize=14)
plt.show
#%%
from sklearn.metrics import mean_squared_error
from math import sqrt
rms=sqrt(mean_squared_error(test,Exp_Smooth.Holt_winter))
print(rms)
#%%
"""
ARIMA auto regressive integrated moving average techinique
ARIMA family-]
I- d
AR-auto regression (p- no of lags)
Ma- moving average model (q-no of lags)
ARMA- combo of ar and ma (p,q)
ARIMA-(p,d,q)
SARIMA- can handel seasonality component
"""
#%% augmented dickey fulloer test
#chaeck stationarity
from statsmodels.tsa.stattools import adfuller
result=adfuller(y)
print('ADF Statistic:',result[0])
print('p-value:%f'%result[1])
#%%
""" 
# if data is non stationary use below test
f=furniture.diff(periods=1)
f.plot(figsize=(10,6))
plt.show()
"""
#%%
import itertools
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('SARIMAX:',pdq[7],'x', seasonal_pdq[7])
#%%
seasonal_pdq
#%% to get values of pdq
aic_list=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:

            mod = sm.tsa.statespace.SARIMAX(y, order=param,
                                            seasonal_order=param_seasonal,enforce_invertibility=False,
                                            enforce_stationarity=False)
            results = mod.fit()
            print('ARIMA',param,'x',param_seasonal,' - AIC:',results.aic)
            aic_list.append(results.aic)
        except:
                continue
#%%
print(min(aic_list))
#%%
mod = sm.tsa.statespace.SARIMAX(y, order=(2,2,1),
                                            seasonal_order=(2,1,0,12),enforce_invertibility=False,
                                            enforce_stationarity=False)
results=mod.fit()
#%%
pred=results.get_prediction(start=pd.to_datetime('2017-01-01'))
plt.figure(figsize=(10,6))
ax=y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax,label='Validation Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Furniture sale')
plt.legend()
plt.show()
#%%%
pred.predicted_mean
#%%
Y_pred=pred.predicted_mean
Y_test=y['2017-01-01':]
from sklearn.metrics import mean_squared_error
from math import sqrt
rms=sqrt(mean_squared_error(Y_test,Y_pred))
print(rms)
#%% forecasting
pred_uc=results.get_forecast(steps=24)
plt.figure(figsize=(10,6))
ax=y['2014':].plot(label='observed')
pred_uc.predicted_mean.plot(ax=ax,label='one-step ahead Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Furniture sale')
plt.legend()
plt.show()
#%%
Y_predictions=pred_uc.predicted_mean
Y_predictions








