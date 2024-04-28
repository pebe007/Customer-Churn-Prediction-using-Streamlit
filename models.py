import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

class RandomForest:
    def __init__(self):
        self.model = None
    
    def tuningParameter(self, input_data, output_data, x_test):
        parameters = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 4, 6, 8, 10],
            'n_estimators': [10, 50, 100, 150, 200]
        }
        RFClass = GridSearchCV(RandomForestClassifier(), param_grid=parameters, scoring='accuracy', cv=5)
        RFClass.fit(input_data, output_data)
        print("Tuned Hyperparameters:", RFClass.best_params_)
        print("Accuracy:", RFClass.best_score_)
        self.model = RandomForestClassifier(**RFClass.best_params_)  # Set the best parameters for the model
        self.model.fit(input_data, output_data)
    
    def predict(self, x_test):
        return self.model.predict(x_test)


class MyXGBoostClassifier:
    def __init__(self):
        self.model = None
    
    def tuningParameter(self, input_data, output_data, x_test):
        parameters = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 150, 200, 250]
        }
        xgb_model = GridSearchCV(xgb.XGBClassifier(), param_grid=parameters, scoring='accuracy', cv=5)
        xgb_model.fit(input_data, output_data)
        print("Tuned Hyperparameters:", xgb_model.best_params_)
        print("Accuracy:", xgb_model.best_score_)
        self.model = xgb.XGBClassifier(**xgb_model.best_params_)
        self.model.fit(input_data, output_data)
    
    def predict(self, x_test):
        return self.model.predict(x_test)