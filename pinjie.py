import pandas as pd
import os
a=[]
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        for i in files:
            #a.append(i)
            a.append(os.path.join(root, i))
df1= pd.read_csv('./sample/mRNA0.txt',sep='\t',header=None)
#df1= pd.read_csv('./sample/methy0.txt',sep='\t',header=None)

file_name('./sample')
for z in a[1:]:
    df2 = pd.read_csv(z,sep='\t',header=None)
    df1=pd.merge(df1,df2,left_on=df1.iloc[:,0], right_on=df2.iloc[:,0], how='left')
    df1=df1.fillna(0)
    df1=df1.drop(['key_0','0_y','1_y'],axis=1)
    df1 = df1.rename(columns={'0_x': 0, '1_x': 1})
    #df1=df1.sort_values(by='1_x',ascending=True)
df1.to_csv('./sample/14patient_feature.txt',sep='\t', header=None,index=0)
