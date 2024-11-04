import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import DecisionTreeClassifier
import os


def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
     
    return



def main():
    root_path = r"yourtraining_data" # change the root path 
    train_X = os.path.join(root_path, "train_x.csv")
    train_y = os.path.join(root_path, "train_y.csv")
    test_X = os.path.join(root_path, "test_x.csv")

    Decision_tree = DecisionTreeClassifier(max_depth=1)
    Decision_tree.fit(train_X,train_y)

    # TODO 
    # build your decision tree
    # predict the output of the testing data
    # remember to save the predict label as .csv file


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

