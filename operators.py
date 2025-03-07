import numpy as np
import pandas as pd
SAMPLE_SIZE=6656
#define Function sets and ternminals 四则算符区
def add(a,b): 
    #print('add') 
    result=np.array(a)+np.array(b)
    #print(np.shape(a),np.shape(b),np.shape(result))
    return  result
def sub(a,b): 
    #print('sub')
    result=np.array(a)-np.array(b)
    #print(np.shape(a),np.shape(b),np.shape(result))
    return result
def mul(a,b): 
    #print('mul')
    result=np.multiply(np.array(a),np.array(b))
    #print(np.shape(a),np.shape(b),np.shape(result))
    return result
def div(a,b): 
    #print("div")
    #print(np.shape(a),np.shape(b))
    if b.all()!=0:
        return np.divide(a,b)
    else:
        return np.full(SAMPLE_SIZE,np.nan)
    
def power(a,b):
    result = np.power(a,b)
    return result


#Time Series Operation 时序算符def区
def ma(x,lookbackwindow=5):
    #print(lookbackwindow)
    if lookbackwindow[0]!=lookbackwindow[1] or lookbackwindow[0]<=0:
        return np.full(len(x),np.nan)
    else:
        #print(lookbackwindow[0])
        result=pd.Series(x).rolling(int(lookbackwindow[0])).mean()
        return np.array(result)
def ema(x,span_window=5):
    #print(span_window)
    if span_window[0]!=span_window[1] or span_window[0]<=1:
        return np.full(len(x),np.nan)

    else:
        result=pd.Series(x).ewm(span=int(span_window[0])).mean()
        return np.array(result)


#High Complexity Functions Defined here
def square(a,b):
    result=np.power(a,2)
    return result

def ts_sum(data,t):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).sum().tolist())
  
    return np.full(SAMPLE_SIZE,value[0])
def sma(data,t):
    window = t
    value = np.nanmean(data,axis=1)
    value = value
    return value
def stddev(data,t):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).std().tolist())
    value = np.nan_to_num(value)
    return value
def ts_rank(data,t):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(10).rank().tolist())
    print(value)
    value = np.full(SAMPLE_SIZE,value[0])
    return value
def ts_min(data,t):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).min().tolist())
    value = np.nan_to_num(value)
    return value
def ts_max(data,t):
    window = 10
    value = np.array(pd.Series(data.flatten()).rolling(window).max().tolist())
    value = np.nan_to_num(value)
    return value
def delta(data,t):
    value = np.diff(data.flatten())
    value = np.append(0, value)
    return value
def delay(data,t):
    period = 1
    value = pd.Series(data.flatten()).shift(1)
    value = np.nan_to_num(value)
    return value
def rank(data,t):
    value = pd.Series(data[-1]).rank(pct=True)
 
    return value
def scale(data,t):
    k = 1
    data = pd.Series(data.flatten())
    value = data.mul(1).div(np.abs(data).sum())
    value = np.nan_to_num(value)
    return value
def ts_argmax(data,t):
    window = 10
    value = pd.Series(data.flatten()).rolling(10).apply(np.argmax) + 1
    value = np.nan_to_num(value)
    return value
def ts_argmin(data,t):
    window = 10
    value = pd.Series(data.flatten()).rolling(10).apply(np.argmin) + 1
    value = np.nan_to_num(value)
    return value




#factor
def mastd(x,t):

       
        ma5=np.nanmean([x[-i-1]for i in range(5)],axis=0)
        ma250=np.nanmean([x[-i-1]for i in range(5)],axis=0)
        ma10=np.nanmean([x[-i-1]for i in range(5)],axis=0)
        ma20=np.nanmean([x[-i-1]for i in range(5)],axis=0)
        ma60=np.nanmean([x[-i-1]for i in range(5)],axis=0)
        ma120=np.nanmean([x[-i-1]for i in range(5)],axis=0)

        
        std=np.nanstd([ma5,ma20,ma10,ma60,ma120,ma250],axis=0)
     
      
        
        return std

TS_FUNCTIONS=[ma,ema,square,rank,stddev,sma]
FUNCTIONS=[add,sub,mul,div,ma,ema,square,rank,stddev,sma,mastd]
