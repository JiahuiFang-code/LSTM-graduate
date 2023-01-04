import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error
plt.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示中文（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False

xt=40#用多少天预测
yt=1#预测多少天

df=pd.read_csv('000895.SZ.csv')
df_X=df[['open', 'high', 'low', 'close', 'vol']]
df_Y=df[[ 'close']]
N=len(df_Y)#数据总长度
# plt.plot(df['open'],label='open')
# plt.plot(df['close'],label='close')
# plt.legend()
# plt.show()
X_data_set = df_X.iloc[:N-yt,:].values#舍去后面yt天，因为这yt天属于不知道
Y_data_set = df_Y.iloc[xt:].values#舍去前xt天，这属于已知的数据，不需要预测
n_all_sample=len(X_data_set)-xt#总样本数
nsep=int(0.8*n_all_sample)#训练集和测试集分割点
print(n_all_sample)

#训练集数据
X_train_set=X_data_set[:nsep+xt]
Y_train_set=Y_data_set[:nsep+yt]

#归一化器
X_scaler = joblib.load('X_scaler')
Y_scaler = joblib.load('Y_scaler')


x_data_set=X_scaler.transform(X_data_set)
y_data_set=Y_scaler.transform(Y_data_set)

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

# 加载已经存在的神经网络模型
model = keras.models.load_model('model.h5')  # 加载模型

# 继续进行神经网络模型训练
#history = model.fit(x_train, y_train, epochs=2000)#训练2000次



# 使用模型
y_train_pre=model.predict(x_train)
Y_train_pre=Y_scaler.inverse_transform(y_train_pre)
Y_train=Y_scaler.inverse_transform(y_train)

y_test_pre=model.predict(x_test)
Y_test_pre=Y_scaler.inverse_transform(y_test_pre)
Y_test=Y_scaler.inverse_transform(y_test)
# print(Y_test)
err=mean_squared_error(Y_test_pre[:,0],Y_test[:,0])
print('测试集mse',err)
#画图
plt.figure()
xa=[_ for _ in range(len(Y_train))]
plt.plot(xa,Y_train,'b-',label='期望值')
plt.plot(xa,Y_train_pre,'r-',label='预测值')
plt.title('训练集')
plt.xlabel('样本')
plt.ylabel('价格')


plt.figure()
xa=[_ for _ in range(len(Y_test))]
plt.plot(xa,Y_test[:,0],'b-',label='期望值')
plt.plot(xa,Y_test_pre[:,0],'r-',label='预测值')
plt.title('测试集')
plt.xlabel('样本')
plt.ylabel('价格')
plt.show()