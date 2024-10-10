###librerias
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler 
import plotly.express as px
import matplotlib.pyplot as plt  # gráficos
from sklearn.metrics import confusion_matrix #### Matriz de confusion 
import seaborn as sns #####Graficos

### Seleccion de variables

def sel_variables(modelos, X, y, threshold):
    var_names_ac = np.array([])
    for modelo in modelos:
        modelo.fit(X, y)
        sel = SelectFromModel(modelo, threshold=threshold, prefit=True)
        var_names = X.columns[sel.get_support()]
        var_names_ac = np.append(var_names_ac, var_names)
    
    var_names_ac = np.unique(var_names_ac)  # Movido fuera del bucle para conservar todas las variables seleccionadas
    
    return var_names_ac

### Medir modelos 

def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["LogisticRegression","RandomForestClassifier","DecisionTreeClassifier"]
    return metric_modelos

