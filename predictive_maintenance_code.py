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
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from streamlit_option_menu import option_menu


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

    
    xgb = XGBClassifier()
    clf = [xgb]
    clf_str = ['XGB']
    xgb_params = {'n_estimators':[100,300,500],
                'max_depth':[5,7,10],
                'learning_rate':[0.01,0.1],
                'objective':['multi:softprob']}
    params = pd.Series(data=[xgb_params],index=clf)

    fitted_models_multi = []
    for model, model_name in zip(clf, clf_str):
        fit_model = tune_and_fit(model,X_train,y_train,params[model],'multi_class')
        fitted_models_multi.append(fit_model)

    task = 'multi_class'
    y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(fitted_models_multi,X_val,y_val,clf_str,task)
    y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(fitted_models_multi,X_test,y_test,clf_str,task)

    # Saving model to file
    best_model = fitted_models_multi[0]
    joblib.dump(best_model, model_filename)

    st.write(f"Trained and saved in " + model_filename)

def predict(csv_file, model_filename):
    st.write("Making predictions...")

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
    #df_pre['Failure_Type'].replace(to_replace=cause_dict, inplace=True)
    df_pre[num_features] = sc.fit_transform(df_pre[num_features]) 

    # Load the saved model
    predictions = model_from_file.predict(df_pre[features])


# Create a bar chart with x-axis showing all failure types
    failure_types = ['Power \nFailure', 'Overstrain \nFailure', 'Heat Dissipation \nFailure', 'Tool Wear \nFailure']
# Calculate the counts of each failure type after making predictions
    failure_counts = [(predictions == 1).sum(), (predictions == 2).sum(), (predictions == 3).sum(), (predictions == 4).sum()]
# Create a bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(failure_types, failure_counts)
# Add labels and a title
    ax.set_xlabel("Failure Types")
    ax.set_ylabel("Count")
    ax.set_title("Failure Types vs. Count")
# Rotate the x-axis labels for better readability (optional)
    #plt.xticks(rotation=45, ha="right")
    plt.xticks(fontsize=8)
# Show the chart using Streamlit
    st.pyplot(fig)

# Calculate the counts of each combination of Type and Failure for the first pie chart
    type_counts_1 = {
        'Type L': ((predictions == 1) & (data['Type'] == 'L')).sum(),
        'Type M': ((predictions == 1) & (data['Type'] == 'M')).sum(),
        'Type H': ((predictions == 1) & (data['Type'] == 'H')).sum()
    }

# Calculate the counts of each combination of Type and Failure for the second pie chart
    type_counts_2 = {
        'Type L': ((predictions == 2) & (data['Type'] == 'L')).sum(),
        'Type M': ((predictions == 2) & (data['Type'] == 'M')).sum(),
        'Type H': ((predictions == 2) & (data['Type'] == 'H')).sum()
    }

    # Filter out categories with 0% from both pie charts
    type_counts_1_filtered = {k: v for k, v in type_counts_1.items() if v != 0}
    type_counts_2_filtered = {k: v for k, v in type_counts_2.items() if v != 0}

# Create two pie charts side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # Create two subplots in one row

# First pie chart
    if type_counts_1_filtered:
        axs[0].pie(type_counts_1_filtered.values(), labels=type_counts_1_filtered.keys(), autopct='%1.1f%%', startangle=90)
        axs[0].set_title("Power failure by type")
    else:
        axs[0].axis('off')  # Turn off the first subplot if no non-zero categories

# Second pie chart
    if type_counts_2_filtered:
        axs[1].pie(type_counts_2_filtered.values(), labels=type_counts_2_filtered.keys(), autopct='%1.1f%%', startangle=90)
        axs[1].set_title("Overstrain failure by type")
    else:
        axs[1].axis('off')  # Turn off the second subplot if no non-zero categories


# Adjust layout for better spacing
    plt.tight_layout()

# Display the chart using Streamlit
    st.pyplot(fig)


#3rd pie chart
    type_counts_3 = {
            'Type L': ((predictions == 3) & (data['Type'] == 'L')).sum(),
            'Type M': ((predictions == 3) & (data['Type'] == 'M')).sum(),
            'Type H': ((predictions == 3) & (data['Type'] == 'H')).sum()
    }
    type_counts_4 = {
            'Type L': ((predictions == 4) & (data['Type'] == 'L')).sum(),
            'Type M': ((predictions == 4) & (data['Type'] == 'M')).sum(),
            'Type H': ((predictions == 4) & (data['Type'] == 'H')).sum()
    }
    type_counts_3_filtered = {k: v for k, v in type_counts_3.items() if v != 0}
    type_counts_4_filtered = {k: v for k, v in type_counts_4.items() if v != 0}
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # Create two subplots in one row
    if type_counts_3_filtered:
        axs[0].pie(type_counts_3_filtered.values(), labels=type_counts_3_filtered.keys(), autopct='%1.1f%%', startangle=90)
        axs[0].set_title("Heat dissipation failure by type")
    else:
        axs[0].axis('off')  # Turn off the first subplot if no non-zero categories
    if type_counts_4_filtered:
        axs[1].pie(type_counts_4_filtered.values(), labels=type_counts_4_filtered.keys(), autopct='%1.1f%%', startangle=90)
        axs[1].set_title("Tool wear failure by type")
    else:
        axs[1].axis('off')  # Turn off the second subplot if no non-zero categories
    plt.tight_layout()
    st.pyplot(fig)



    st.write("Power Failure :")
    power_failure = data[predictions == 1]
    st.write(power_failure)

    st.write("Overstrain failure :")
    overstrain_failure = data[predictions == 2]
    st.write(overstrain_failure)

    st.write("Heat dissipation failure :")
    heat_dissipation_failure = data[predictions == 3]
    st.write(heat_dissipation_failure)

    st.write("Tool Wear failure :")
    tool_wear_failure = data[predictions == 4]
    st.write(tool_wear_failure)

def list_saved_models():
    saved_models = [f[:-4] for f in os.listdir() if f.endswith(".pkl")]
    return saved_models

st.markdown(
    """
    <style>
    /* Define CSS for the logo */
    .logo-container {
        position:absolute;
        top: -150px; /* Adjust the top position as needed */ /*-150px*/
        left: -310px; /* Adjust the left position as needed */
        /*z-index:1;*/
    }
    

    /* Add your logo image */
    .logo-image {
        width: 120px; /* Adjust the width as needed */
        height: 85px; /* Maintain aspect ratio */
    }

    
    </style>
    """
, unsafe_allow_html=True)


st.title("Predictive Maintenance Model App")

st.markdown(
    """
    <div class="logo-container">
        <img src="https://www.3i-infotech.com/wp-content/uploads/2021/06/h_logo_color.png" class="logo-image">
    </div>
    
    """
, unsafe_allow_html=True)

page = option_menu(
    menu_title = None,
    options=["Train Model","Predict"],
    icons = ["check-circle","check-circle-fill"],
    default_index=0,
    orientation="horizontal",
)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if page == "Train Model":
    st.header("Train AI Model")
    uploaded_file = st.file_uploader("Upload a CSV file for training:", type=["csv"])
    #model_filename = st.text_input("Enter the model filename (without extension):")
    model_filename = st.text_input(
        "Enter the model filename",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder="Enter the filename without extension",
    )

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
                predict(uploaded_file, selected_model+".pkl")
    else:
        st.header("No models trained yet")
