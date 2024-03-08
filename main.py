import pandas as pd
import numpy as np

columns = ['sepal_length','sepal_width','petal_length','petal_width','type']

df = pd.read_csv("iris.csv",skiprows= 0, header=None,names=columns)
print(df)


class Node():
    def __init__(self, feature_index = None,threshold = None, left = None, right = None, info_gain = None,value = None):


        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # Leaf Node
        self.value = value


class DecisionTreeClassifier():
    def __init__(self,min_samples_split=2, max_depth=2):

        self.root = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self,dataset,curr_depth=0):
        X,Y = dataset[:,:-1],dataset[:,-1]

        num_samples, num_features = np.shape(X)
        if num_samples > self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset,num_samples,num_features)
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"],curr_depth+1)

                right_subtree = self.build_tree(best_split["dataset_right"],curr_depth+1)

                return Node(best_split["feature_index"], best_split["threshold"],left_subtree,right_subtree,best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)