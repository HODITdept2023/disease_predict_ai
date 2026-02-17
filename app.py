# ==========================================================
# M.TECH MULTI-DISEASE DIAGNOSTIC SYSTEM - STREAMLIT APP
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ----------------------------------------------------------
# LOGIN SYSTEM
# ----------------------------------------------------------

def login():
    st.title("🔐 Medical AI Login")

    username = st.text_input("User ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "csdmed2026ai":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid Credentials")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# ----------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------

st.title("🩺 Multi-Disease Diagnostic System (M.Tech Project)")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("mtech_disease_dataset_5000.csv")

df = load_data()

X = df.drop("Disease", axis=1)
y = df["Disease"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Initialize Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(kernel='linear', probability=True)
}

# Train Models
for model in models.values():
    model.fit(X_train, y_train)

# ----------------------------------------------------------
# SYMPTOM DROPDOWN SECTION
# ----------------------------------------------------------

st.header("📝 Enter Patient Symptoms")

symptom_list = sorted(list(X.columns))

symptom1 = st.selectbox("Symptom 1", [""] + symptom_list)
symptom2 = st.selectbox("Symptom 2", [""] + symptom_list)
symptom3 = st.selectbox("Symptom 3", [""] + symptom_list)
symptom4 = st.selectbox("Symptom 4", [""] + symptom_list)
symptom5 = st.selectbox("Symptom 5", [""] + symptom_list)

if st.button("Predict Disease"):

    selected_symptoms = [s for s in [symptom1, symptom2, symptom3, symptom4, symptom5] if s != ""]

    input_vector = [0] * len(X.columns)

    for symptom in selected_symptoms:
        index = list(X.columns).index(symptom)
        input_vector[index] = 1

    input_array = np.array([input_vector])

    st.subheader("🔍 Prediction Results")

    for name, model in models.items():
        pred = model.predict(input_array)
        disease_name = le.inverse_transform(pred)
        st.write(f"**{name} predicts:** {disease_name[0]}")

    # ------------------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------------------

    st.subheader("📊 Model Performance")

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    # Accuracy Bar Chart
    fig1, ax1 = plt.subplots()
    ax1.bar(results.keys(), results.values())
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy Comparison")
    st.pyplot(fig1)

    # Confusion Matrix (Random Forest)
    best_model = models["Random Forest"]
    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)

    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, cmap="Blues", ax=ax2)
    ax2.set_title("Confusion Matrix - Random Forest")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
    y_score = best_model.predict_proba(X_test)

    fig3, ax3 = plt.subplots()

    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

    ax3.plot([0,1],[0,1],'--')
    ax3.set_title("Multi-Class ROC Curve")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend(fontsize=6)

    st.pyplot(fig3)
