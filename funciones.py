# Selecciona modelos 
def sel_variables(modelos, X, y, threshold):
    
    var_names_ac = np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit = True, threshold = threshold)
        var_names = modelo.feature_names_in_[sel.get_support()]
        var_names_ac = np.append(var_names_ac, var_names)
        var_names_ac = np.unique(var_names_ac)
    
    return var_names_ac


# Validación del rendimiento de los modelos 
def medir_modelos(modelos, scoring, X, y, cv):

    metric_modelos = pd.DataFrame()
    for modelo in modelos:
        scores = cross_val_score(modelo, X, y, scoring = scoring, cv = cv )
        pdscores = pd.DataFrame(scores)
        metric_modelos = pd.concat([metric_modelos,pdscores], axis = 1)
    
    metric_modelos.columns = ["logistic_r","rf_classifier","sgd_classifier","xgboost_classifier"]
    return metric_modelos