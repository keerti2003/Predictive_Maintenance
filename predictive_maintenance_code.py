import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import streamlit as st
from imblearn.over_sampling import SMOTENC


def eval_preds(model,X,y_true,y_pred,task):
    if task == 'binary':
        y_true = y_true['Target']
        cm = confusion_matrix(y_true, y_pred)
        proba = model.predict_proba(X)[:,1]
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f2 = fbeta_score(y_true, y_pred, pos_label=1, beta=2)
    elif task == 'multi_class':
        y_true = y_true['Failure_Type']
        cm = confusion_matrix(y_true, y_pred)
        proba = model.predict_proba(X)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba, multi_class='ovr', average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    metrics = pd.Series(data={'ACC':acc, 'AUC':auc, 'F1':f1, 'F2':f2})
    metrics = round(metrics,3)
    return cm, metrics

def tune_and_fit(clf,X,y,params,task):
    if task=='binary':
        f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Target'])
    elif task=='multi_class':
        f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Failure_Type'])
        
    train_time = time.time()-start_time
    mins = int(train_time//60)
    return grid_model

def predict_and_evaluate(fitted_models,X,y_true,clf_str,task):
    cm_dict = {key: np.nan for key in clf_str}
    metrics = pd.DataFrame(columns=clf_str)
    y_pred = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models,clf_str):
        y_pred[model_name] = fit_model.predict(X)
        if task == 'binary':
            cm, scores = eval_preds(fit_model,X,y_true,y_pred[model_name],task)
        elif task == 'multi_class':
            cm, scores = eval_preds(fit_model,X,y_true,y_pred[model_name],task)
        cm_dict[model_name] = cm
        metrics[model_name] = scores
    return y_pred, cm_dict, metrics

def train_model(csv_file,model_filename):
    st.write("Training the AI model...")
    data = pd.read_csv(csv_file)

    data['Tool_wear'] = data['Tool_wear'].astype('float64')
    data['Rotational_speed'] = data['Rotational_speed'].astype('float64')

    df = data.copy()
    df.drop(columns=['UDI','ProductID'], inplace=True)

    features = [col for col in df.columns if df[col].dtype=='float64' or col =='Type']
    target = ['Target','Failure_Type']
    idx_RNF = df.loc[df['Failure_Type']=='Random Failures '].index
    
    df.drop(index=idx_RNF, inplace=True)

    idx_ambiguous = df.loc[(df['Target']==1) & (df['Failure_Type']=='No Failure ')].index
   

    df.drop(index=idx_ambiguous, inplace=True)

    
    df.reset_index(drop=True, inplace=True)   # Reset index
    n = df.shape[0]
    
    num_features = [feature for feature in features if df[feature].dtype=='float64']
    
    n_working = df['Failure_Type'].value_counts()['No Failure ']
    desired_length = round(n_working/0.8)
    spc = round((desired_length-n_working)/4)  #samples per class
    balance_cause = {'No Failure ':n_working,
                    'Overstrain Failure ':spc,
                    'Heat Dissipation Failure ':spc,
                    'Power Failure ':spc,
                    'Tool Wear Failure ':spc}
    sm = SMOTENC(categorical_features=[0,7], sampling_strategy=balance_cause, random_state=0)
    df_res, y_res = sm.fit_resample(df, df['Failure_Type'])
    
    sc = StandardScaler()
    type_dict = {'L': 0, 'M': 1, 'H': 2}
    cause_dict = {'No Failure ': 0,
                'Power Failure ': 1,
                'Overstrain Failure ': 2,
                'Heat Dissipation Failure ': 3,
                'Tool Wear Failure ': 4}
    df_pre = df_res.copy()
    df_pre['Type'].replace(to_replace=type_dict, inplace=True)
    df_pre['Failure_Type'].replace(to_replace=cause_dict, inplace=True)
    df_pre[num_features] = sc.fit_transform(df_pre[num_features]) 
    
    X, y = df_pre[features], df_pre[['Target','Failure_Type']]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=df_pre['Failure_Type'], random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, stratify=y_trainval['Failure_Type'], random_state=0)

    lr = LogisticRegression(random_state=0,multi_class='ovr')
    knn = KNeighborsClassifier()
    svc = SVC(decision_function_shape='ovr')
    rfc = RandomForestClassifier()
    xgb = XGBClassifier()
    clf = [lr,knn,svc,rfc,xgb]
    clf_str = ['LR','KNN','SVC','RFC','XGB']
    lr_params = {'random_state':[0]}
    knn_params = {'n_neighbors':[1,3,5,8,10]}
    svc_params = {'C': [1, 10, 100],
                'gamma': [0.1,1],
                'kernel': ['rbf'],
                'probability':[True],
                'random_state':[0]}
    rfc_params = {'n_estimators':[100,300,500,700],
                'max_depth':[5,7,10],
                'random_state':[0]}
    xgb_params = {'n_estimators':[100,300,500],
                'max_depth':[5,7,10],
                'learning_rate':[0.01,0.1],
                'objective':['multi:softprob']}
    params = pd.Series(data=[lr_params,knn_params,svc_params,rfc_params,xgb_params],index=clf)


    fitted_models_multi = []
    for model, model_name in zip(clf, clf_str):
        fit_model = tune_and_fit(model,X_train,y_train,params[model],'multi_class')
        fitted_models_multi.append(fit_model)

    task = 'multi_class'
    y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(fitted_models_multi,X_val,y_val,clf_str,task)
    y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(fitted_models_multi,X_test,y_test,clf_str,task)
    metrics_final = metrics_val*metrics_test
    # Calculating best model
    macc = 0
    bestm = 0
    for i, j in enumerate(clf_str):
        acc = metrics_final[j][0]
        if acc > macc:
            macc = acc
            bestm = i
    print(bestm, "\t", clf_str[bestm])

    # Saving model to file
    best_model = fitted_models_multi[bestm]
    joblib.dump(best_model, model_filename)

    st.write(f"Trained and saved in " + model_filename)

def predict(csv_file, model_filename):
    st.write("Making predictions...")

    # Load the saved model
    model_from_file = joblib.load(model_filename)
    data = pd.read_csv(csv_file)

    data['Tool_wear'] = data['Tool_wear'].astype('float64')
    data['Rotational_speed'] = data['Rotational_speed'].astype('float64')
    
    df = data.copy()
    df.drop(columns=['UDI','ProductID'], inplace=True)

    features = [col for col in df.columns if df[col].dtype=='float64' or col =='Type']
    num_features = [feature for feature in features if df[feature].dtype=='float64']

    sc = StandardScaler()
    type_dict = {'L': 0, 'M': 1, 'H': 2}
    cause_dict = {'No Failure ': 0,
                'Power Failure ': 1,
                'Overstrain Failure ': 2,
                'Heat Dissipation Failure ': 3,
                'Tool Wear Failure ': 4}

    df_pre = df.copy()
    df_pre['Type'].replace(to_replace=type_dict, inplace=True)
    df_pre[num_features] = sc.fit_transform(df_pre[num_features]) 

    # Load the saved model
    predictions = model_from_file.predict(df_pre[features])

    st.write("Non-Failure Rows:")
    non_failure_rows = data[predictions == 0]
    st.write(non_failure_rows)

    st.write("Power Failure Rows:")
    power_failure = data[predictions == 1]
    st.write(power_failure)

    st.write("Overstrain failure Rows:")
    overstrain_failure = data[predictions == 2]
    st.write(overstrain_failure)

    st.write("Heat dissipation failure Rows:")
    heat_dissipation_failure = data[predictions == 3]
    st.write(heat_dissipation_failure)

    st.write("Tool Wear failure Rows:")
    tool_wear_failure = data[predictions == 4]
    st.write(tool_wear_failure)

def list_saved_models():
    saved_models = [f for f in os.listdir() if f.endswith(".pkl")]
    return saved_models

# Define the Streamlit app
st.title("Predictive Maintenance Model App")
# Add a sidebar for navigation
page = st.sidebar.selectbox("Select a page:", ["Train Model", "Predict"])

if page == "Train Model":
    st.header("Train AI Model")
    uploaded_file = st.file_uploader("Upload a CSV file for training:", type=["csv"])
    model_filename = st.text_input("Enter the model filename (without extension):")
    if st.button("Submit"):
        if uploaded_file is not None:
            with st.spinner("Training in progress..."):
                train_model(uploaded_file, model_filename+".pkl")
            st.success("Training completed!")
elif page == "Predict":
    st.header("Make Predictions")
    uploaded_file = st.file_uploader("Upload a CSV file for predictions:", type=["csv"])
    saved_models = list_saved_models()
    if saved_models is not None:
        selected_model = st.selectbox("Select a trained model:", saved_models)
        if st.button("Submit"):
            if uploaded_file is not None:
                predict(uploaded_file, selected_model)
    else:
        st.header("No models trained yet")
