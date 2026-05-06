import numpy as np
import pandas as pd

#this is a decision tree class
#you can choose different methods such as Gain or Gini for deciding the best split
#you can also choose different method to prune the tree 
#X is the feature matrix and y is the target vector

Data = pd.read_excel('../data/watermelon.xlsx')
print(Data)
print(Data.columns)
print(Data.columns.size)
print(Data.iloc[:,1])
print(Data.iloc[:,1].value_counts())

type_num = Data.iloc[:,-1].value_counts()
print(type_num)
print(type_num[0])
print(type_num[1])
print(type_num.size)
print(type_num.index)
print(Data.shape[0])

class DescionTree:
    def __init__(self,method,prune_method):
        self.tree = None
        self.method = method
        self.prune_method = prune_method
           
    def _create_tree_(self,data):
        # if prune_method == 'forward':
        #     #forward prune
        #     pass
        # elif prune_method == 'backward':
        #     #backward prune
        #     pass
        feature_num = data.columns.size-1
        labels = data.iloc[:,-1].value_counts()
        if labels.size == 1:
            return labels.index[0]
        if feature_num == 0:
            return labels.idxmax()
        feature_tmp,val_tmp,val_dict_tmp = self.method(data)
        tree = {feature_tmp:{}}
        for value in data[feature_tmp].value_counts().index:
            mask_i = data[feature_tmp]==value
            data_i = data[mask_i].drop(columns=feature_tmp)
            if data_i.shape[0] == 0:
                tree[feature_tmp][value]=labels.idxmax()
            else: 
                tree[feature_tmp][value]=self._create_tree_(data_i)        
        return tree
    
    def fit(self,X,y):
        data = pd.concat([X,y],axis=1)
        self.tree = self._create_tree_(data)
        return self.tree
    
    def _predict_(self,tree,sample):
        if not isinstance(tree,dict):
            return tree
        feature = next(iter(tree))
        feature_value = sample[feature]
        if feature_value in tree[feature]:
            subtree = tree[feature][feature_value]
            return self._predict_(subtree,sample)
        else:
            subtrees = tree[feature].values()
            leaves = []
            for subtree in subtrees:
                leaves.extend(collect_leaves(subtree))
            if leaves:
                return pd.Series(leaves).value_counts().idxmax()
            else:
                return None
        
    def predict(self,X):
        # y_pred = []
        # for i in range(X.shape[0]):
        #     sample = X.iloc[i,:]
        #     pred = self._predict_(self.tree,sample)
        #     y_pred.append(pred)
        # return np.array(y_pred)
        y_pred = X.apply(lambda row: self._predict_(self.tree, row), axis=1).to_numpy()
        return y_pred
        
def collect_leaves(tree):
    if not isinstance(tree, dict):
        return [tree]
    leaves = []
    for subtree in tree.values():
        leaves.extend(collect_leaves(subtree))
    return leaves

def entropy(data_tmp):
    data_num = data_tmp.shape[0]
    type_num = data_tmp.iloc[:,-1].value_counts()
    # for i in range(type_num.size):
    #     ci = type_num[i]/data_num * np.log2(type_num[i]/data_num)
    #     entropy_now -= ci
    p = type_num/data_num
    entropy = -np.sum(p*np.log2(p+1e-9))
    return entropy

def Gain(data_tmp, ratio_tag = True):
    feature_num = data_tmp.columns.size-1
    entropy_now = entropy(data_tmp)
    gain_dict = {}
    for i in range(1,feature_num):
        current_feature = data_tmp.iloc[:,i]
        current_feature_type = current_feature.value_counts()
        data_sub_num = current_feature_type.size
        split_entropy = 0
        for j in range(data_sub_num):
            mask_j = data_tmp.iloc[:,i]==current_feature_type.index[j]
            data_sub_j = data_tmp[mask_j]
            entropy_sub = entropy(data_sub_j)
            data_num = data_tmp.shape[0]
            j_num = data_sub_j.shape[0]
            weighted_entropy = j_num/data_num * entropy_sub
            split_entropy += weighted_entropy
        if ratio_tag == False:
            gain_dict[data_tmp.columns[i]] = entropy_now - split_entropy
        else:
            split_info = -np.sum((current_feature_type/data_num)*np.log2(current_feature_type/data_num+1e-9))
            gain_dict[data_tmp.columns[i]] = (entropy_now - split_entropy)/split_info
    best_feature_name= max(gain_dict, key=gain_dict.get)
    best_gain = gain_dict[best_feature_name] 
    return best_feature_name,best_gain,gain_dict

def Gini_index(data_tmp):
    data_num = data_tmp.shape[0]
    type_num = data_tmp.iloc[:,-1].value_counts()
    p = type_num/data_num
    gini = 1 - np.sum(p**2)
    return gini

def Gini(data_tmp):
    feature_num = data_tmp.columns.size-1
    gini_dict = {}
    for i in range(1,feature_num):
        current_feature = data_tmp.iloc[:,i]
        current_feature_type = current_feature.value_counts()
        data_sub_num = current_feature_type.size
        split_weighted_gini = 0
        for j in range(data_sub_num):
            mask_j = data_tmp.iloc[:,i]==current_feature_type.index[j]
            data_sub_j = data_tmp[mask_j]
            gini_sub = Gini_index(data_sub_j)
            data_num = data_tmp.shape[0]
            j_num = data_sub_j.shape[0]
            split_weighted_gini += j_num/data_num * gini_sub
        gini_dict[data_tmp.columns[i]] = split_weighted_gini
    best_feature_name= min(gini_dict, key=gini_dict.get)
    best_gini = gini_dict[best_feature_name]
    return best_feature_name,best_gini,gini_dict
    
    
    

