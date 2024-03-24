import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import pickle

def evaluvation_metrics(x_test_path,y_test_path,model):

    #Read data
    X=pd.read_csv(x_test_path)
    y_target=pd.read_csv(y_test_path)
    #seperate numerical and categorical columns
    numerical_columns=X.select_dtypes(exclude='object')
    
                       
    # numerical-- Scalling
    model_scaling=load('Model\standard_scaler.pkl')
    scaled_data=model_scaling.transform(numerical_columns)
    numerical_scaled_data=pd.DataFrame(scaled_data,columns=numerical_columns.columns)

    Outcome = numerical_scaled_data


    # model testing
    log_reg=load(model)
    y_pred=pd.DataFrame(log_reg.predict(Outcome))
    test_score=accuracy_score(y_target, y_pred)*100
    
    
    return y_pred, test_score
