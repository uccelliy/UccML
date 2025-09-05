import numpy as np
import pandas as pd

#this is a decision tree class
#you can choose different methods such as Gain or Gini for deciding the best split
#you can also choose different method to prune the tree 
#X is the feature matrix and y is the target vector

Data = pd.read_excel('西瓜.xlsx')
input,output = Data.iloc[:,1:-1],Data.iloc[:,-1]

def Gain(table_now):
    gain = 0
    return gain

def Gini(table_now):
    gini = 0
    return gini

#首先对于表来说要计算当前的熵,然后对于每个特征计算信息增益,选择信息增益最大的特征作为分裂节点,根据分裂属性生成新的表
def CreateTree(input,output,method=Gini, prune_method='forward'):
    input=np.array(input)
    output=np.array(output)
    
    
    if prune_method == 'forward':
        #forward prune
        pass
    elif prune_method == 'backward':
        #backward prune
        pass
    return None
    
    

