import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm #Scipy的stats模块包含了多种概率分布的随机变量，norm正态分布
from sklearn.preprocessing import StandardScaler #preprocessing数据预处理，StandardScaler数据归一化/标准化
from scipy import stats
import warnings
warnings.filterwarnings('ignore')#抑制第三方警告
%matplotlib inline

#读取训练集数据
train_data = pd.read_csv('F:/桌面/房价预测/train.csv')
train_data.columns

#对销售价格的描述统计
train_data['SalePrice'].describe()#返回描述性统计
#柱状图
sns.distplot(train_data['SalePrice']);
#偏度skewness和峰度kurtosis
print('偏度:%f'% train_data['SalePrice'].skew())
print('峰度:%f'% train_data['SalePrice'].kurt())

#探索价格于变量的关系
#地上居住面积和价格
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));#ylim设置y轴范围
#地下室面积和价格
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));

#价格与属性特征的关系
#整体材料和表面质量与价格的关系
var = 'OverallQual'
data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)#concat数据重塑与联结
f, ax =  plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
#价格与建筑年份的关系
var = 'YearBuilt'
data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
f, ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000);
plt.xticks(rotation=90);

#相关性热力图
corrmat = train_data.corr()
f,ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True);
#销售价格相关矩阵热力图
k = 10#热力图的变量数
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)#返回泊松矩阵的相关系数
sns.set(font_scale=1.25)#字号大小
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#价格与相关变量的散点图
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(train_data[cols], size=2.5)
plt.show();

#缺失数据
total = train_data.isnull().sum().sort_values(ascending=False)#计算缺失值并按数量排序
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)#sort_values排序，ascending降序
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)
#处理丢失数据
train_data = train_data.drop((missing_data[missing_data['Total']>1]).index,1)
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
train_data.isnull().sum().max()#检查还有没有缺失数据

#单变量分析
#标准化数据
saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);#fit_transform数据拟合后再标准化
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]#拟合后底部10个数据
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]#拟合后顶部10个数据
print('底部范围的分布：')
print(low_range)
print('\n顶部范围的分布：')
print(high_range)

#双变量分析
#价格和居住面积的分析
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#删除异常值
train_data.sort_values(by='GrLivArea',ascending=False)[:2]
train_data = train_data.drop(train_data[train_data['Id']==1299].index)
train_data = train_data.drop(train_data[train_data['Id']==524].index)
#价格与地下室总面积
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#正态分布分析
#价格直方图和正态概率图
sns.distplot(train_data['SalePrice'],fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
#对数变换
train_data['SalePrice'] = np.log(train_data['SalePrice'])
#转换的直方图和正态概率图
sns.distplot(train_data['SalePrice'],fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'],plot=plt)
#居住面积的直方图和正态概率图
sns.distplot(train_data['GrLivArea'],fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['GrLivArea'],plot=plt)
#对数变换
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
#转换后的直方图和正态概率图
sns.distplot(train_data['GrLivArea'],fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['GrLivArea'],plot=plt)
#地下室面积的直方图和正态概率图
sns.distplot(train_data['TotalBsmtSF'],fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['TotalBsmtSF'],plot=plt)
#创建新变量
#如果地下室面积>0，则赋值1，如果地下室面积==0，则赋值0
train_data['HasBsmt'] = pd.Series(len(train_data['TotalBsmtSF']), index=train_data.index)
train_data['HasBsmt'] = 0
train_data.loc[train_data['TotalBsmtSF']>0,'HasBsmt'] = 1
#对数变换
train_data.loc[train_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_data['TotalBsmtSF'])
#转换后的直方图和正态概率图
sns.distplot(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'],fit=norm);
fig = plt.figure()
res = stats.probplot(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'],plot=plt)

#价格和居住面积的同方差分析
plt.scatter(train_data['GrLivArea'],train_data['SalePrice']);
#价格和地下室居住面积同方差分析
plt.scatter(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], train_data[train_data['TotalBsmtSF']>0]['SalePrice']);
#将类别变量转变为虚拟变量
train_data = pd.get_dummies(train_data)

#建立模型进行预测

