
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import mean_absolute_error
from scipy import stats
import seaborn as sns
from math_function import sigmoid_function,maxminNorm,Z_Norm

# data = pandas.read_csv('Features.csv')
# X = data[['K','T_up','T_tol','slope','eata','re_hight','mid_hight']]
X_data = loadmat('Features_full.mat')
X_tem = X_data['Features_full']
X = np.array((X_tem[:,24],X_tem[:,19],X_tem[:,3],X_tem[:,0],X_tem[:,20])).T
print(type(X)," ",X.shape)

# 数据标准化并未降低mae
# Z标准化,mae=0.3075,  0-1标准化，mae = 0.3075
X_norm = maxminNorm(X)
X_norm1 = X_norm[0]
X_norm2 = X_norm[1]
X_norm3 = X_norm[2]
X_norm4 = X_norm[3]
X_norm5 = X_norm[4]
#X_norm6 = X_norm[5]
X_normal = np.c_[X_norm1,X_norm2,X_norm3,X_norm4,X_norm5]
# sns.distplot(X_norm1)

# 仅用Box变换，mae = 0.2951
converted_x1 = stats.boxcox(X[:,0])[0]
converted_x2 = stats.boxcox(X[:,1])[0]
converted_x3 = stats.boxcox(X[:,2])[0]
converted_x4 = stats.boxcox(X[:,3])[0]
converted_x5 = stats.boxcox(X[:,4])[0]
converted_X = np.c_[converted_x1,converted_x2,converted_x3,converted_x4,converted_x5]
# sns.distplot(converted_x1)





# Y = data['gan']
Y_data = loadmat('TG_Label.mat')
Y = Y_data['TG_Label']
print(type(Y)," ",Y.shape)
# sns.distplot(Y)

#对标签进行BOX-COX变换
# converted_Y = stats.boxcox(Y)[0] 
# sns.distplot(converted_data1)





# 数据集制作
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=6,8,9,11,16,17);  # 默认3：1划分
# ('K','slope','time','tol_s','ER1','ER2','ER3',)  random_state=3,8,9,10,11,12,16,17,18,
# 'K','slope','time','tol_s','ER1','ER2','ER3','fr_HR','fr_max'， 2，3，8，9，11，12，16
X_train,X_test,Y_train,Y_test = train_test_split(X_normal,Y,test_size=0.3,random_state=2);  # 默认3：1划分
print(X_train.shape," ",X_test.shape," ",Y_train.shape," ",Y_test.shape)

# 线性回归
lrg = LinearRegression();
model = lrg.fit(X_train,Y_train);
print(model)
print(lrg.intercept_)  # 输出截距
'''
coef = zip(['K','slope','time','tol_s','ER1','ER2','ER3','fr_HR','fr_max'],lrg.coef_)  # 特征和系数对应
for T in coef :
    print(T);   # 输出系数
'''
# 预测
Y_predit = lrg.predict(X_test);
print(Y_predit)


# 决定系数评价指标
from sklearn.metrics import r2_score
R2 =  r2_score(Y_test, Y_predit)
print('确定系数R2',R2)

'''
sum_mean = 0;
for i in range(len(Y_pred)):
    sum_mean+=(Y_pred[i]-Y_test[i])**2
sum_erro = np.sqrt(sum_mean/len(Y_pred));
print('均方误差',sum_erro);

sum_abs = 0;
for i in range(len(Y_pred)):
    sum_abs+=abs(Y_pred[i]-Y_test[i]);
mean_abs = (sum_abs/len(Y_pred));
print('平均绝对误差',mean_abs);

var_s2 = 0;
for i in range(len(Y_pred)):
    var_s2+=(Y_pred[i]-np.mean(Y_test))**2
mean_s2 = (var_s2/(len(Y_pred)-1))
print('离散度',mean_s2);

sum_mean = 0;
for i in range(len(Y_predit)):
    sum_mean+=(Y_predit[i]-Y_test[i])**2;
sum_erro = np.sqrt(sum_mean/len(Y_predit));
print('均方误差',sum_erro);



# 决定系数评价指标
from sklearn.metrics import r2_score
R2 =  r2_score(Y_test, Y_predit)
print('确定系数R2',R2)

sum_abs = 0;
for i in range(len(Y_predit)):
    sum_abs+=abs(Y_predit[i]-Y_test[i]);
mean_abs = (sum_abs/len(Y_predit));
print('平均绝对误差',mean_abs);

var_s2 = 0;
for i in range(len(Y_predit)):
    var_s2+=(Y_predit[i]-np.mean(Y_test))**2
mean_s2 = (var_s2/(len(Y_predit)-1))
print('离散度',mean_s2);

# 若误差<0.3,则认为正确，计算准确率
acc_idx = 0;
for i in range(len(Y_predit)):
    ac=abs(Y_predit[i]-Y_test[i])
   # print('ac',ac)
    
    if ac<=0.3:
        acc_idx = acc_idx+1

print('准确率',acc_idx/len(Y_predit));

# 绘制曲线
plt.figure();
plt.plot(range(len(Y_predit)),Y_predit,'--',color='red',linewidth=0.5,label='predict')
plt.plot(range(len(Y_predit)),Y_predit,'*',mfc='none',color='red')
plt.plot(range(len(Y_test)),Y_test,'-',color='black',linewidth=0.5,label='test')
plt.plot(range(len(Y_test)),Y_test,'o',mfc='none',color='black')

plt.xlabel('Gan')
plt.ylabel('Value')
plt.legend();
plt.show();
'''
fig=plt.figure()
plt.plot(np.arange(len(Y_test)), Y_test,'go-',label='reference TG')
plt.plot(np.arange(len(Y_predit)),Y_predit,'ro-',label='predicted TG')
plt.xlabel('sample')
plt.ylabel('TG(mmol/L)')
plt.legend()
plt.show()

# calculate error\n",
mae = mean_absolute_error(Y_test, Y_predit)
std = np.std(np.abs(Y_test-Y_predit))
PCC=stats.pearsonr(Y_test[:,0],Y_predit[:,0])[0]
print('MAE:', mae)
print('STD:', std)
print('PCC:', PCC)