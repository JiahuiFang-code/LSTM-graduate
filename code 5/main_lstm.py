import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import joblib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error,mean_absolute_error
plt.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示中文（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False


xt=40#用多少天预测
yt=1#预测多少天

df=pd.read_csv('000895.SZ.csv')
df_X=df[['open', 'high', 'low', 'close', 'vol']]
df_Y=df[[ 'close']]
N=len(df_Y)#数据总长度

X_data_set = df_X.iloc[:N-yt,:].values#舍去后面yt天，因为这yt天属于不知道
Y_data_set = df_Y.iloc[xt:].values#舍去前xt天，这属于已知的数据，不需要预测
n_all_sample=len(X_data_set)-xt+1#总样本数
nsep=int(0.8*n_all_sample)#训练集和测试集分割点

#训练集数据
X_train_set=X_data_set[:nsep+xt]
Y_train_set=Y_data_set[:nsep+yt]

#归一化器
X_scaler = MinMaxScaler().fit(X_train_set)
Y_scaler = MinMaxScaler().fit(Y_train_set)
joblib.dump(X_scaler,'X_scaler')#保存归一化器
joblib.dump(Y_scaler,'Y_scaler')#保存归一化器

x_data_set=X_scaler.transform(X_data_set)
y_data_set=Y_scaler.transform(Y_data_set)
print(x_data_set.shape,X_data_set.shape)


X,Y=[],[]
for i in range(n_all_sample):
    x_temp=x_data_set[i:i+xt,:]
    y_temp = y_data_set[i:i + yt, :].reshape((yt,))
    X.append(x_temp)
    Y.append(y_temp)


# 用以LSTM的测试集和训练集划分
x_train=np.array(X[:nsep])
x_test=np.array(X[nsep:])
y_train=np.array(Y[:nsep])
y_test=np.array(Y[nsep:])
print(x_train.shape)
print(y_train.shape)

# 神经网络模型搭建
model = Sequential()
model.add(LSTM(units = 40,  activation='relu',
               return_sequences = False , input_shape = (xt, 5)))
# model.add(Dropout(0.3))
# model.add(LSTM(units = 30,  activation='relu',
#               return_sequences = False ))
# model.add(Dropout(0.3))#dropout 正则化
# model.add(Dense(10,activation='relu'))
model.add(Dense(yt,activation='linear',kernel_regularizer=regularizers.l1(0.001)))#,kernel_regularizer=regularizers.l1(0.001)正则化层的添加，权重参数w添加L1正则化；
##  bias_regularizer=regularizers.l2(0.01),#在偏置向量b添加L2正则化
##  activity_regularizer=regularizers.l1_l2(0.01),#在输出部分添加L1和L2结合的正则化


#模型编译
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(learning_rate=0.001))

# 神经网络模型训练
history = model.fit(x_train, y_train, epochs=500)#训练2000次
#模型保存
model.save('model.h5')



y_train_pre=model.predict(x_train)
Y_train_pre=Y_scaler.inverse_transform(y_train_pre)
Y_train=Y_scaler.inverse_transform(y_train)

y_test_pre=model.predict(x_test)
Y_test_pre=Y_scaler.inverse_transform(y_test_pre)
Y_test=Y_scaler.inverse_transform(y_test)

# print(Y_test_pre)
# print(Y_test)

# 计算测试集上的mse
# err=mean_squared_error(Y_test_pre[:,0],Y_test[:,0])
# print('测试集mse',err)

# 保存数据
# df1=pd.DataFrame(Y_test_pre,columns=[1,2,3,4,5,6,7])
# df1.to_excel('测试集预测结果.xlsx',index=None)
# df1=pd.DataFrame(Y_test,columns=[1,2,3,4,5,6,7])
# df1.to_excel('测试集真实结果.xlsx',index=None)

plt.figure()
xa=[_ for _ in range(len(Y_train))]
plt.plot(xa,Y_train,'b-',label='期望值')
plt.plot(xa,Y_train_pre,'r-',label='预测值')
plt.title('训练集')
plt.xlabel('样本')
plt.ylabel('价格')

#画图
plt.figure()
tt=[_ for _ in range(len(history.history['loss']))]
plt.plot(tt,history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')


plt.figure()
xa=[_ for _ in range(len(Y_test))]
plt.plot(xa,Y_test[:,0],'b-',label='期望值')
plt.plot(xa,Y_test_pre[:,0],'r-',label='预测值')
plt.title('测试集')
plt.xlabel('样本')
plt.ylabel('价格')
plt.show()