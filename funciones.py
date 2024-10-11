# Librerías para manipulación de datos
import pandas as pd
import numpy as np

# Librerías para visualización
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Librerías para machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Librerías para la coneción con bigquery
from google.cloud import bigquery
from google.oauth2 import service_account




#Regresión logística
def regresionLogistica (df,y):
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    
    X_train, X_test,y_train,y_test = train_test_split(df,y,test_size=0.2, random_state=42)

    X_train_std = X_train
    X_test_std = X_test

    lr = LogisticRegression(max_iter=1000,class_weight="balanced", random_state=42).fit(X_train_std,y_train)
    y_pred_train = lr.predict(X_train_std)
    y_pred_test = lr.predict(X_test_std)
    y_pred_prob_train = lr.predict_proba(X_train_std)
    y_pred_prob_test = lr.predict_proba(X_test_std)
    mc_train=confusion_matrix(y_train,y_pred_train)
    mc_test=confusion_matrix(y_test,y_pred_test)
    tn, fp, fn, tp = mc_train.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print('-'*30,'TRAIN','-'*30)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Especificidad: {especificidad}')
    print(f'F1 score: {f1_score}')
    print('Train score: ',lr.score(X_train_std,y_train))

    tn, fp, fn, tp = mc_test.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    accuracy = lr.score(X_test_std,y_test)
    print('-'*30,'TEST','-'*30)
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'Especificidad : {especificidad}')
    print(f'F1 score : {f1_score}')
    print('Test score: ',accuracy)
    
    resultados = {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'y_pred_train' : y_pred_train,
        'y_pred_test' : y_pred_test,
        'y_pred_prob_train' : y_pred_prob_train,
        'y_pred_prob_test' : y_pred_prob_test,
        'precision' : precision,
        'recall' : recall,
        'especificidad' : especificidad,
        'f1_score' : f1_score,
        'accuracy' : accuracy
    }
    
    return resultados, lr

#Bosque aleatorio clasificador
def bosqueAleatorio (df,y):
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    
    X_train, X_test,y_train,y_test = train_test_split(df,y,test_size=0.2, random_state=42)


    ranfor=RandomForestClassifier(n_estimators=300,
                                  max_depth=30,
                                  n_jobs=-1,
                                  max_leaf_nodes=20,
                                  min_samples_leaf=10,
                                  class_weight="balanced", random_state=42).fit(X_train,y_train)
    y_pred_train=ranfor.predict(X_train)
    y_pred_test=ranfor.predict(X_test)
    y_pred_prob_train=ranfor.predict_proba(X_train)
    y_pred_prob_test=ranfor.predict_proba(X_test)
    mc_train=confusion_matrix(y_train,y_pred_train)
    mc_test=confusion_matrix(y_test,y_pred_test)
    tn, fp, fn, tp = mc_train.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print('-'*30,'TRAIN','-'*30)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Especificidad: {especificidad}')
    print(f'F1 score: {f1_score}')
    print('Train score: ',ranfor.score(X_train,y_train))

    tn, fp, fn, tp = mc_test.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    accuracy = ranfor.score(X_test,y_test)
    print('-'*30,'TEST','-'*30)
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'Especificidad : {especificidad}')
    print(f'F1 score : {f1_score}')
    print('Test score: ',accuracy)
    
    resultados = {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'y_pred_train' : y_pred_train,
        'y_pred_test' : y_pred_test,
        'y_pred_prob_train' : y_pred_prob_train,
        'y_pred_prob_test' : y_pred_prob_test,
        'precision' : precision,
        'recall' : recall,
        'especificidad' : especificidad,
        'f1_score' : f1_score,
        'accuracy' : accuracy
    }
    
    return resultados, ranfor