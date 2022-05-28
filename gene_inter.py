
import pandas as pd
import numpy as np
#根据RF构建的网络，进行筛边
df1=pd.read_csv('./HumanMethylation_pair.txt',sep='\t',index_col=0)
#df1=pd.read_csv('./mRNA_d_pair.txt',sep='\t',index_col=0)
df=df1.where(np.triu(np.ones(df1.shape)).astype(np.bool))
df=df.stack().reset_index()
df.columns = ['Row','Column','Value']
df=df.drop(df[df['Value'].isin([1])].index)#删除指定元素的行
df=df[df['Value']>0.8]#0.8
df=df.drop(['Value'],axis=1)
df.to_csv('./pair/methy_gene_pair.txt',sep='\t', header=None, index=0)


