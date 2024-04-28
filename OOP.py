import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
import xgboost as xgb
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
        self.column_medians = {}
        self.column_modes = {}

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data = self.data.drop_duplicates()
        self.data = self.data.drop(['Unnamed: 0', 'id', 'CustomerId', 'Surname'], axis=1)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

        for col in self.input_df.columns:
            if self.input_df[col].dtype == 'float64' or self.input_df[col].dtype == 'int64':
                self.column_medians[col] = self.input_df[col].median()
            else:
                self.column_modes[col] = self.input_df[col].mode()[0]
    
    def fill_na_with_dict(self, fill_dict):
        for col, value in fill_dict.items():
            if col in self.input_df.columns:
                self.input_df[col].fillna(value, inplace=True)
            else:
                print(f"Column '{col}' not found in dataframe.")
    
    def encode_categorical(self):
        categorical_cols = self.input_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.input_df[col] = pd.factorize(self.input_df[col])[0]
    


class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train, self.x_test, self.y_train, self.y_test = [None] * 4
        self.robust = None
        self.recall_score = None
    
    def RobustScaling(self):
        self.robust = RobustScaler()
        self.robust.fit_transform(self.x_train)  
        self.robust.transform(self.x_test)
    
    def split_data(self, test_size=0.2, random_state=42): #split test data dengan test size 20% dari original data and original state = 42
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
    
    def ClassificationReport(self,y_pred):
        print(classification_report(self.y_test, y_pred))
    
    def Recall_score(self,y_pred):
        self.recall_score = recall_score(self.y_test, y_pred)
    
def prediction(model,x_test):
        return model.predict(x_test)
  
class RandomForest:
    def __init__(self):
        self.y_pred = None
        self.criterion = None
        self.max_depth = None
        self.n_estimators = None
        self.model = RandomForestClassifier()
        
    #melakukan secara automatis tidak seperti modelling file 
    def tuningParameter(self,input_data,output_data,x_test):
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
        self.model.fit(input_data,output_data)
        self.y_pred = self.model.predict(x_test)
    
        
class XGBoostClassifier:
    def __init__(self):
        self.y_pred = None
        self.max_depth = None
        self.learning_rate = None
        self.n_estimators = None
        self.model = xgb.XGBClassifier()

    #melakukan secara automatis tidak seperti modelling file 
    def tuningParameter(self,input_data,output_data,x_test):
        parameters = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 150, 200, 250]
        }
        xgb_model = GridSearchCV(xgb.XGBClassifier(), param_grid=parameters, scoring='accuracy', cv=5)
        xgb_model.fit(input_data, output_data)
        print("Tuned Hyperparameters:", xgb_model.best_params_)
        print("Accuracy:", xgb_model.best_score_)
        self.model = xgb.XGBClassifier(**xgb_model.best_params_)  # Set the best parameters for the model
        self.y_pred = xgb_model.predict(x_test)
    
file_path = 'data_B.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('churn')
data_handler.fill_na_with_dict(data_handler.column_medians)
data_handler.fill_na_with_dict(data_handler.column_modes)
data_handler.encode_categorical()  # Encode categorical variables

# Check if missing values are filled and categorical variables are encoded
# print("Missing values filled with column medians and modes:")
# print(data_handler.input_df.head())

model_handler = ModelHandler(data_handler.input_df, data_handler.output_df)
model_handler.split_data()
model_handler.RobustScaling()

random_forest = RandomForest()
random_forest.tuningParameter(model_handler.x_train,model_handler.y_train,model_handler.x_test)
random_forest_model = model_handler.ClassificationReport(random_forest.y_pred)
random_forest_recall = model_handler.recall_score(random_forest.y_pred)


xgboost = XGBoostClassifier()
xgboost.tuningParameter(model_handler.x_train,model_handler.y_train,model_handler.x_test)
xg_boost_model = model_handler.ClassificationReport(xgboost.y_pred)
xg_boost_recall = model_handler.Recall_score(xgboost.y_pred)


if(xg_boost_recall>=random_forest_recall):
    model_used = model_xgb
else:
    model_used = model_GridSearch

model_used.fit(model_handler.x_test,model_handler.y_test)


with open('model.pickle', 'wb') as dump_var:
    pickle.dump(model_used, dump_var)
    
pickle_in = open('model.pickle', 'rb')
pickle_model = pickle.load(pickle_in)