from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN,OPTICS
from lifelines.statistics import logrank_test
import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.manifold import TSNE
def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED


df=pd.read_csv('./kirc/patient_feature.txt',sep='\t',header=None)
df=df.set_index(0)
df=df.drop([1],axis=1)


# for i in range(8,80):
#     for j in range(2,10):
#         db = OPTICS(eps=i, min_samples=j,cluster_method='dbscan').fit(df)
#         labels = db.labels_
#         if len(set(list(labels)))==2:
#             result = Counter(labels)
#             print(i,j,len(set(list(labels))),result)


dd = OPTICS(eps=41, min_samples=2,cluster_method='dbscan').fit(df)
df['cluster_db'] = dd.labels_



df=df.sort_values(by='cluster_db')
df=df.reset_index()
data_zs=df.iloc[:,1:1281]
tsne=TSNE()
tsne.fit_transform(data_zs)

tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index)
Euclidean_dis=EuclideanDistances(tsne.values,tsne.values)
d1=pd.DataFrame(Euclidean_dis)
d1=-d1
f,ax=plt.subplots(figsize=(5,4))
sns.heatmap(d1, cmap = 'Blues', ax = ax, vmin=-10, vmax=0,cbar = False)
ax.set_title('THCA',fontsize=15)
ax.set_ylim([len(d1), 0])
plt.xticks([])
plt.yticks([])
plt.show()

data=pd.read_csv('./TCGA_Clinical/kirc/Clinical/nationwidechildrens.org_clinical_patient_kirc.txt', sep='\t')
data=data[["bcr_patient_barcode","gender","vital_status","last_contact_days_to","death_days_to"]]
data=data.drop(data[data['last_contact_days_to'].isin(['[Completed]'])].index)#删除指定元素的行
t=data.merge(df,left_on=data["bcr_patient_barcode"],right_on=df[0],how='left')
class_mapping = {'Dead':0, 'Alive':1}
t["vital_status"] = t["vital_status"].map(class_mapping)
del t["key_0"]
del t[0]
t=t.dropna(axis=0,how='any')

dataset55=t[["vital_status","last_contact_days_to","death_days_to","cluster_db"]].sort_values('cluster_db')
t4=dataset55[dataset55['cluster_db']==-1]
t4.loc[t4.last_contact_days_to=='[Not Available]','last_contact_days_to']=0
t4.loc[t4.last_contact_days_to=='[Not Applicable]','last_contact_days_to']=0
t4.loc[t4.death_days_to=='[Not Available]','death_days_to']=0
t4.loc[t4.death_days_to=='[Not Applicable]','death_days_to']=0
aaa33=t4.astype(int)
aaa33['time']=aaa33['last_contact_days_to']+aaa33['death_days_to']
T2 = aaa33['time']
E2 = aaa33['vital_status']
T0=T2
E0=E2
kmf00 = KaplanMeierFitter()
kmf00.fit(T2.astype('int'), event_observed=E2.astype('int')) # more succiently, kmf.fit(T,E)
kmf00.plot()
plt.show()

t5=dataset55[dataset55['cluster_db']==0]
t5.loc[t5.last_contact_days_to=='[Not Available]','last_contact_days_to']=0
t5.loc[t5.last_contact_days_to=='[Not Applicable]','last_contact_days_to']=0
t5.loc[t5.death_days_to=='[Not Available]','death_days_to']=0
t5.loc[t5.death_days_to=='[Not Applicable]','death_days_to']=0
aaa33=t5.astype(int)
aaa33['time']=aaa33['last_contact_days_to']+aaa33['death_days_to']


T2 = aaa33['time']
E2 = aaa33['vital_status']
T1=T2
E1=E2
kmf11 = KaplanMeierFitter()
kmf11.fit(T2.astype('int'), event_observed=E2.astype('int')) # more succiently, kmf.fit(T,E)
kmf11.plot()
plt.show()



result1=logrank_test(T1,T0, event_observed_A=E1, event_observed_B=E0)
print(result1.p_value)
h1=kmf00.survival_function_
h2=kmf11.survival_function_

x1=h1.index
y1=h1
x2=h2.index
y2=h2

plt.plot(x1,y1,color='green', label='Subtype I',linewidth=2,marker='.')
plt.plot(x2,y2,color='red',label='Subtype II',linewidth=2,marker='.')#red
plt.legend(loc='best',frameon=False)
plt.text(1500,0.95 ,"THCA",size=18)
plt.xlabel('Time(days)')
plt.ylabel('Survival Probability',)
plt.text(2800, 0.3, r'$p=0.03$',size=13)
plt.tick_params(labelsize=8)
plt.show()



