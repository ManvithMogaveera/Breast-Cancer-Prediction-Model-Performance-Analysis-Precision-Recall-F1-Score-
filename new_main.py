import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc,ConfusionMatrixDisplay
)
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score,RandomizedSearchCV


MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def builded_pipeline(num_attrib):
    num_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attrib)
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    original_data = pd.read_csv('data.csv')
    diagnosis_column = {'M':0,'B':1}
    original_data['diagnosis'] = original_data['diagnosis'].map(diagnosis_column)
    original_data = original_data.drop('id',axis=1)

    # print(original_data['radius_se'].describe())
    shuffler = StratifiedShuffleSplit(n_splits=1,random_state=42,test_size=0.2)
    original_data['radius_se_cat'] = pd.cut(original_data['radius_se'],bins=[0,0.2,0.4,0.6,1,np.inf],labels=[1,2,3,4,5])
    for train_set,test_set in shuffler.split(original_data,original_data['radius_se_cat']):
        strata_train = original_data.loc[train_set]
        strata_test = original_data.loc[test_set]
    train_data = strata_train.copy()
    X_train = train_data.drop('diagnosis',axis=1)
    Y_train = train_data['diagnosis']
    X_test = strata_test.drop('diagnosis',axis=1)
    Y_test = strata_test['diagnosis']

    num_attrib = X_train.columns.tolist()
    new_pipeline = builded_pipeline(num_attrib)
    transformed_data = new_pipeline.fit_transform(X_train)
    model = LGBMClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=2,n_estimators=200)
    Y_pred =model.fit(transformed_data,Y_train)
    Y_pred_2 = model.predict(new_pipeline.transform(X_test))
    cm = confusion_matrix(Y_test,Y_pred_2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)    
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix_bcp.png')
    plt.show()
    
    y_pred_prob = model.predict_proba(new_pipeline.transform(X_test))[:,1]
    fpr,tpr,thresholds = roc_curve(Y_test,y_pred_prob)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC = {roc_auc:.2f}") 
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate ")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve ")
    plt.legend(loc = "lower right")
    plt.savefig("roc_curve_bcp.png")
    plt.show()
    print(f"Classification Report:{classification_report(Y_test,Y_pred_2)}") 
    joblib.dump(model,MODEL_FILE)
    joblib.dump(new_pipeline,PIPELINE_FILE)
    print('YOUR PIPELINE IS READY ENJOY!!!')

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv("input.csv")
    data_transformed = pipeline.transform(input_data)
    predicted_output = model.predict(data_transformed)
    input_data['diagnosis'] = predicted_output
    input_data.to_csv("prediction.csv",index=False)
    print("CONGRATULATIONS!!!The inference was a success and the prediction is saved in prediction.csv kindly check it out.")