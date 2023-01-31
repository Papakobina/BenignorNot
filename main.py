import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
# train test split
from sklearn.model_selection import train_test_split
# import the data from sklearn 
from sklearn.datasets import load_breast_cancer
# grid search implementation to comapre to regular svm with default values
from sklearn.model_selection import GridSearchCV


# load the data
cancer_dataset = load_breast_cancer()

# convert the data into a dataframe
data_frame_cancer_dataset = pd.DataFrame(cancer_dataset['data'], columns=cancer_dataset['feature_names'])

# train test split
x = data_frame_cancer_dataset
y = cancer_dataset['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)

class Train_model:
    def __init__(self):
        pass

    def train_svm_model(self):
        # svm implementation with default values
        Svm_model = SVC()
        final_model = Svm_model.fit(x_train, y_train)
        return final_model

    def results_of_svm_model(self, model):
        prediction_of_model = model.predict(x_test)         
        print(confusion_matrix(y_test, prediction_of_model))
        print(classification_report(y_test, prediction_of_model))        

    def train_grid_implementation(self):
        param_grid = {
            'C':[0.1,1,10,100,1000],
            'gamma': [1, 0.1,0.01,0.001,0.0001]
        }
        grid = GridSearchCV(SVC(), param_grid, verbose=3)
        final_model = grid.fit(x_train,y_train)
        return final_model

    def results_of_grid_implementation(self, model):
        grid_predictions = model.predict(x_test)
        print(confusion_matrix(y_test, grid_predictions))
        print(classification_report(y_test, grid_predictions))


run_models = Train_model()

svm_model = run_models.train_svm_model()
run_models.results_of_svm_model(svm_model)
grid_implementation_for_model = run_models.train_grid_implementation()
run_models.results_of_grid_implementation(grid_implementation_for_model)
      




# the grid search implementation was a plus as the acuracy improved. 