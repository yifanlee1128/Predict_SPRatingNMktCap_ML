import pandas as pd
import numpy as np
import math
import missingno as msno
from scipy import stats
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet,LogisticRegression,ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC,LinearSVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


df = pd.read_csv("Raw Data Set.txt",delimiter='\t')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

StringParaNameList=['Ticker', 'Name', 'S&P Rating', 'country', 'industry']

def toNum(x):
    try:
        return float(x)
    except ValueError:
        return np.nan
for (columnName, columnData) in df.iteritems():
    if columnName not in StringParaNameList:
        newData=[toNum(x) for x in columnData.values]
        df[columnName]=newData
df.dropna(subset=['Mkt cap'],inplace=True)
df.dropna(thresh=20,inplace=True)

df_RatingTransResult = pd.read_excel("RatingTransferResult.xlsx")
df['N S&P Rating']=df_RatingTransResult.values

df_NewIndustry = pd.read_excel("NewIndustryCategory.xlsx")
df['industry']=df_NewIndustry.values

msno.matrix(df).figure

IndustryRelated=['sales 1','fcf','ltm sales gr','sales gr','sales 3yr avg growth','ret on cap','capex','ebitda/int','debt/t. bv','cfo/debt',
                'debt/ebitda','cfo/interest','ebitda/int.1','ebit/int','ebit/sales','debt/t asts','cfo/capex']
IndustryList=list(set(df['industry'].values))

group_df=df.groupby('industry')
for IndustryName in IndustryList:
    temp_df=group_df.get_group(IndustryName)
    for FactorName in IndustryRelated:
        temp_df[FactorName].fillna(temp_df[FactorName].median(),inplace=True)
    df.update(temp_df)

df['foreign scr'].fillna(0,inplace=True)

#process and get the corresponding company size in excel
#df['Mkt cap'].to_csv("Mkt cap.csv")
df['Mkt cap rank']=pd.read_excel("Mkt Cap Ranking.xlsx").values

IndustryNCapRelated=['ebitda','ltm ebitda gr','ltm ni gr','ebitda gr','ni growth','ni 5yr gr','debt','cash int','interest','ebit']
for IndustryName in IndustryList:
    for rank in ['L','M','S']:
        temp=df[df['industry']==IndustryName].groupby('Mkt cap rank').get_group(rank)
        for factorName in IndustryNCapRelated:
            temp[factorName].fillna(temp[factorName].median(),inplace=True)
        df.update(temp)

df.dropna(inplace=True)
Y_Rating=df['S&P Rating']
#capital N means numerical
Y_N_Rating=df['N S&P Rating']
df.drop(['S&P Rating','Ticker','Name','Mkt cap rank','country','industry','N S&P Rating'],axis=1,inplace=True)
msno.matrix(df).figure


temp_boolean=(np.abs(stats.zscore(df)) < 3).all(axis=1)
df=df[temp_boolean]
Y_Rating=Y_Rating[temp_boolean]
Y_N_Rating=Y_N_Rating[temp_boolean]


transformer = RobustScaler().fit(df)
scaled_df_for_PCA=pd.DataFrame(transformer.transform(df),columns=df.columns)

scaled_Y_MktCap=scaled_df_for_PCA['Mkt cap']
scaled_Y_Rating=Y_Rating
scaled_Y_N_Rating=Y_N_Rating
scaled_df_for_PCA.drop(['Mkt cap'],axis=1,inplace=True)

scaled_df_for_PCA.corr(method='spearman')

scaled_df_for_PCA.drop(['sales 2','ebitda/int.1','ebit','tang assets'],axis=1,inplace=True)


pca = PCA()
pca.fit(scaled_df_for_PCA)
print(pca.explained_variance_ratio_)

weight=np.square(pca.components_)
contribution=[]
for list in weight.T:
    contribution.append(np.dot(list,pca.explained_variance_ratio_))
print(contribution)

#cumulative contribution
import copy
temp_contribution=copy.deepcopy(contribution)
temp_contribution.sort(reverse=True)
print(np.cumsum(temp_contribution))

index=np.argsort(contribution)
index=index[::-1]
FinalVarName_PCA=[]
for i in range(12):
    FinalVarName_PCA.append(scaled_df_for_PCA.columns[index[i]])
print(FinalVarName_PCA)


model = ElasticNet()
rfe1 = RFE(model,12)
fit1 = rfe1.fit(scaled_df_for_PCA, scaled_Y_N_Rating)

#factor ranking with response variable "S&P Rating"
print(fit1.ranking_)

#factor ranking with response variable "Mkt cap"
rfe2 = RFE(model,12)
fit2 = rfe2.fit(scaled_df_for_PCA, scaled_Y_MktCap)
print(fit2.ranking_)

FinalVarName_RFE_ForRating=[]
for i in range(len(fit1.ranking_)):
    if fit1.ranking_[i]==1:
        FinalVarName_RFE_ForRating.append(scaled_df_for_PCA.columns[i])
FinalVarName_ForRating=sorted(set(FinalVarName_RFE_ForRating+FinalVarName_PCA))
print(FinalVarName_ForRating)

FinalVarName_RFE_ForCap=[]
for i in range(len(fit2.ranking_)):
    if fit2.ranking_[i]==1:
        FinalVarName_RFE_ForCap.append(scaled_df_for_PCA.columns[i])

FinalVarName_ForCap=sorted(set(FinalVarName_RFE_ForCap+FinalVarName_PCA))
print(FinalVarName_ForCap)


transformer2 = StandardScaler().fit(df)
scaled_df_for_Forecast=pd.DataFrame(transformer2.transform(df),columns=df.columns)
scaled_Y_MktCap=scaled_df_for_Forecast['Mkt cap']
scaled_df_for_Forecast.drop(['Mkt cap'],axis=1,inplace=True)
scaled_df_for_Forecast.drop(['sales 2','ebitda/int.1','ebit','tang assets'],axis=1,inplace=True)

X_forRating=scaled_df_for_Forecast[FinalVarName_ForRating]


kf = KFold(n_splits=5,shuffle=True,random_state=3)
kf.get_n_splits(X_forRating)


model_Name_List=["LinearSVC","LogisticRegression","MLPClassifier","RandomForestClassifier"]
modelList=[LinearSVC(),LogisticRegression(),MLPClassifier(max_iter=800),RandomForestClassifier()]

for model,modelName in zip(modelList,model_Name_List):
    i=1
    temp_Result=pd.DataFrame()
    print(modelName)
    for train_index, test_index in kf.split(X_forRating):
        model.fit(X_forRating.iloc[train_index], scaled_Y_Rating.iloc[train_index])
        print(model.score(X_forRating.iloc[test_index], scaled_Y_Rating.iloc[test_index]))
        temp_Result["Pred "+str(i)]=model.predict(X_forRating.iloc[test_index])
        temp_Result["Real "+str(i)]=scaled_Y_Rating.iloc[test_index].values
        #for i,j in zip(scaled_Y_Rating.iloc[test_index],model.predict(X_forRating.iloc[test_index])):
            #print(i,j)
        i=i+1
    temp_Result.to_csv("TestResult_"+modelName+".csv")


plt.clf()
print("LinearSVC Chart Result")
img1=mpimg.imread('SVC_Group1.png')
img2=mpimg.imread('SVC_Group2.png')
img3=mpimg.imread('SVC_Group3.png')
img4=mpimg.imread('SVC_Group4.png')
img5=mpimg.imread('SVC_Group5.png')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(30,10))
p1 = ax1.imshow(img1)
p2 = ax2.imshow(img2)
p3 = ax3.imshow(img3)
p4 = ax4.imshow(img4)
p5 = ax5.imshow(img5)
plt.show()

print("LogisticRegression Chart Result")
img1=mpimg.imread('LR_Group1.png')
img2=mpimg.imread('LR_Group2.png')
img3=mpimg.imread('LR_Group3.png')
img4=mpimg.imread('LR_Group4.png')
img5=mpimg.imread('LR_Group5.png')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(30,10))
p1 = ax1.imshow(img1)
p2 = ax2.imshow(img2)
p3 = ax3.imshow(img3)
p4 = ax4.imshow(img4)
p5 = ax5.imshow(img5)
plt.show()

plt.close()
print("MLPClassifier Chart Result")
img1=mpimg.imread('MLP_Group1.png')
img2=mpimg.imread('MLP_Group2.png')
img3=mpimg.imread('MLP_Group3.png')
img4=mpimg.imread('MLP_Group4.png')
img5=mpimg.imread('MLP_Group5.png')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(30,10))
p1 = ax1.imshow(img1)
p2 = ax2.imshow(img2)
p3 = ax3.imshow(img3)
p4 = ax4.imshow(img4)
p5 = ax5.imshow(img5)
plt.show()

plt.close()
print("RandomForestClassifier Chart Result")
img1=mpimg.imread('RF_Group1.png')
img2=mpimg.imread('RF_Group2.png')
img3=mpimg.imread('RF_Group3.png')
img4=mpimg.imread('RF_Group4.png')
img5=mpimg.imread('RF_Group5.png')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(30,10))
p1 = ax1.imshow(img1)
p2 = ax2.imshow(img2)
p3 = ax3.imshow(img3)
p4 = ax4.imshow(img4)
p5 = ax5.imshow(img5)
plt.show()

#get the corresponding factors
X_forCap=scaled_df_for_Forecast[FinalVarName_ForCap]


model_Name_List=["ElasticNetCV","MLPRegressor","LinearSVR","RandomForestRegressor"]
modelList=[ElasticNetCV(),MLPRegressor(max_iter=800),LinearSVR(),RandomForestRegressor()]

for model,modelName in zip(modelList,model_Name_List):
    i=1
    temp_Result=pd.DataFrame()
    print(modelName)
    for train_index, test_index in kf.split(X_forCap):
        model.fit(X_forCap.iloc[train_index], scaled_Y_MktCap.iloc[train_index])
        print(model.score(X_forCap.iloc[test_index], scaled_Y_MktCap.iloc[test_index]))
        temp_Result["Pred "+str(i)]=model.predict(X_forCap.iloc[test_index])
        temp_Result["Real "+str(i)]=scaled_Y_MktCap.iloc[test_index].values
        #for i,j in zip(scaled_Y_Rating.iloc[test_index],model.predict(X_forRating.iloc[test_index])):
            #print(i,j)
        i=i+1
    temp_Result.to_csv("Cap_TestResult_"+modelName+".csv")

plt.close()
print("Result for predicting MktCap")
img=mpimg.imread('Cap_SumResult.png')
plt.figure(figsize = (15,15))
imgplot = plt.imshow(img)
plt.show()