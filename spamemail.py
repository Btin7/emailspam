import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.title("📧 Spam Email Detection")

st.write("Classify emails as spam or not spam using machine learning")

df = pd.read_csv("emails.csv", encoding='latin-1')
df = df[['text', 'spam']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

with st.expander("Dataset"):
    st.write(df)

with st.expander("Data Analysis"):
    st.subheader("Class Distribution")
    counts = df['label'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, palette="muted", ax=ax)
    ax.set_xticklabels(['Not Spam', 'Spam'])
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Message Length Distribution")
    df['message_length'] = df['message'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='message_length', hue='label', bins=30, kde=True, palette='muted', ax=ax)
    ax.set_xlabel("Message Length")
    st.pyplot(fig)

tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X = tfidf.fit_transform(df['message'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

with st.expander("Model Comparison"):
    st.subheader("Model Accuracy")
    fig, ax = plt.subplots()
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="muted", ax=ax)
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

    best_model_name = max(results, key=results.get)
    st.write(f"Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.2f}")

    st.subheader("Model Evaluation Metrics")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, models[best_model_name].predict_proba(X_test)[:, 1])

    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    st.write(f"ROC-AUC: {roc_auc:.2f}")

if st.button("Hyperparameter Tuning for Random Forest"):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    st.write(f"Best Parameters: {best_params}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    st.write(f"Accuracy with Best Parameters: {accuracy_score(y_test, y_pred):.2f}")

st.sidebar.header("Input Email")
email_input = st.sidebar.text_area("Enter email content")
if st.sidebar.button("Classify"):
    best_model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2)
    best_model.fit(X_train, y_train)
    input_transformed = tfidf.transform([email_input])
    prediction = best_model.predict(input_transformed)
    label = "Spam" if prediction[0] == 1 else "Not Spam"
    st.sidebar.success(f"Prediction: {label}")
