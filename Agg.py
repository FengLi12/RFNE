import os
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
df=pd.read_csv('./emb/methy_emb.txt',sep=' ',header=None,index_col=0)
dd = AgglomerativeClustering(linkage='ward', n_clusters=5).fit(df)#euclidean
df['cluster'] = dd.labels_
df=df.sort_values(by='cluster')
df1=pd.read_csv('./gene_mutation/gene_mutation.txt',header=None,sep='\t')
df1=df1.drop(df1[df1[2].isin([10])].index)#删除指定元素的行
df1=df1.drop_duplicates(subset=[0,1,2],keep='first')
for i in range(0,5):
    df2=df[df['cluster']==i]
    df2=df2.reset_index()
    df12 = pd.merge(df1 , df2 , left_on=df1.iloc[: , 1] , right_on=df2.iloc[: , 0] , how='inner')
    df12 = df12.drop(['key_0' , '1_x'] , axis=1)
    l = [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 11 , 12 , 13 , 14 , 15]
    jj = 1
    df12 = df12.set_index('0_x')
    for j in l:
        d = df12[(df12['2_x'] == j) & (df12['cluster']==i)]
        p = d['0_y'].drop_duplicates(keep='first')
        for z in p:
            s = (df12['0_y'] == z).sum().sum()
            dd = d[d['0_y'] == z]
            s1 = (dd['0_y'] == z).sum().sum()
            D = dd * (s1 / s)
            if jj == 1:
                DD = D
                DD['label'] = j
                jj = 0
            else:
                D['label'] = j
                DD = pd.concat([DD , D])
        print(j)

    data = DD.reset_index()
    data = data.groupby([data['0_x'] , data['label']]).sum()  # 按列A求和
    data = data.drop(['0_y' , '2_x','cluster'] , axis=1)
    if i==0:
        data.to_csv('./sample/mRNA0.txt' , header=None , sep='\t')
    elif i == 1:
        data.to_csv('./sample/mRNA1.txt' , header=None , sep='\t')
    elif i == 2:
        data.to_csv('./sample/mRNA2.txt' , header=None , sep='\t')
    elif i==3:
        data.to_csv('./sample/mRNA3.txt' , header=None , sep='\t')
    elif i==4:
        data.to_csv('./sample/mRNA4.txt' , header=None , sep='\t')
    else:
        data.to_csv('./sample/mRNA5.txt' , header=None , sep='\t')