import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.sidebar.header("Upload your CSV flie")
uploadedFile = st.sidebar.file_uploader("Upload a  csv file", type=["csv"])

if uploadedFile is not None:
    data = pd.read_csv(uploadedFile)

    st.title("Titanic Survivors Prediction APP")

    if "data" in locals() or "data" in globals():
        st.header("Raw dataset")
        isCheck = st.checkbox("Show raw dataset")
        if isCheck:
            st.write(data)

    importanceFeatures = data.columns.drop(["PassengerId", "Name", "Ticket",
                                            "Cabin", "Survived"]).tolist()

    st.header("Select Features for Prediction")
    selectedFeatures = st.multiselect("Select features to use for predction:",
                                      options = importanceFeatures, default =["pclass", "Sex", "Age", "Fare", "Embarked"])

    if "selectedFeatures" not in st.session_state:
        st.session_state["selectedFeatures"] = selectedFeatures
    if selectedFeatures != st.seesion_state['seletedFeatures']:
        del st.session_state["gridBestModel"]
        st.session_state["gridBestModel"] = selectedFeatures

    data = data[selectedFeatures + ["Survived"]].dropna()

    data = pd.get_dummies(data, drop_first=True)
    st.write(data)

    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    
    st.header("Hyperparameter Tuning with GridSearchCV")
    paramGrid = {
        "n_estimators" : [50, 100, 150],
        "max_depth" : [None, 10, 20],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4]
    }

    if "gridBestModel" not in st.session_state:
        gridSearch = GridSearchCV(RandomForestClassifier(random_state=42),
                                  paramGrid, cv=3, n_jobs=-1, verbose=2)
        gridSearch.fit(X_train, y_train)
        st.session_state["gridBestModel"] = gridSearch.best_estimator_

    gridBestModel = st.session_state["gridBestModel"]

    y_pred = gridBestModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the best model: {accuracy * 100:.2f}%")

    st.header("Feature Importance")
    featureImportances = np.sort(gridBestModel.feature_importances_)[::-1]
    featureImportances = featureImportances.reshape (1, -1)
    featureImportances = pd.DataFrame (featureImportances,
                                       columns=X. columns)

    fig, ax = plt.subplots()
    sns.barplot(x=featureImportances.iloc[0, :],
                y=featureImportances.columns, ax=ax)
    ax.set_title("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    st. header('Predict Survival')
    userInput = {}
    for feature in selectedFeatures:
        if feature == "Pclass":
            userInput[feature] = st.radio(
                "Select Pclass",
                options=[1,2,3],
                index=0,
                horizontal=True
            )
        elif feature in ['Fare', 'Age']:
            userInput [feature] = st.slider(
                f"Enter {feature}",
                min_value=float(data[feature].min()),
                max_value=float (data[feature].max()),
                value=float(data[feature].mean()),
            )
        elif feature in ['SibSp', 'Parch']:
            userInput[feature] = st.slider(
                f"Enter {feature}",
                min_value=int(data[feature].min()) ,
                max_value=int(data[feature].max()),
                value=int(data[feature].mean()),
            )

        elif feature == "Sex":
            userInput["Sex_male"] = st.selectbox("Select Gender",
                                                 options=['Male', 'Female']) == 'Male'
        elif feature == "Embarked":
            userInput[feature] = st.selectbox("Select Embarkation Port",
                                                 options=['C', 'Q', 'S'])
            if userInput[feature] == 'Q':
                userInput["Embarked_Q"] = 1
                userInput["Embarked_S"] = 0
            elif userInput[feature] == 'S':
                userInput["Embarked_Q"] = 0
                userInput["Embarked_S"] = 1
            else:
                userInput["Embarked_Q"] = 0
                userInput["Embarked_S"] = 0

        inputDF = pd.DataFrame([userInput])
        if "Embarked" in inputDF.columns:
            inputDF = inputDF.drop(columns=["Embarked"], axis=1, errors='ignore')

        inputArray = np.array(inputDF.iloc[0, :])
        inputArray = inputArray.reshape (1, -1)

        if st.button('Predict'):
            prediction = gridBestModel.predict(inputArray)[0]
            if prediction == 1:
                st. success('The passenger is likely to survive!')
            else:
                st.error('The passenger is unlikely to survive.')

    else:
        st.error("No dataset available. Please upload a CSV file.")
        

