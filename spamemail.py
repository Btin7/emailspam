import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


st.title('📧 Spam Email Detection')

st.write('Email detection made by Btin7')

with st.expander("Load Dataset"):
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]  # Selecting relevant columns
    df.columns = ['label', 'message']  # Renaming columns for clarity

    st.write('**Dataset Preview**')
    st.dataframe(df.head())

    st.write('**Dataset Shape:**', df.shape)

with st.expander("Data Preparation"):
    st.write('**Target Mapping**')
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encode labels (0 = ham, 1 = spam)
    st.write(df['label'].value_counts())


    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    st.write('Data is prepared and split into training and testing sets.')

with st.expander("Train Model"):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_vectorized, y_train)

    y_pred = clf.predict(X_test_vectorized)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {acc:.2f}")


with st.sidebar:
    st.header("Input Email Content")
    email_input = st.text_area(
        "Type or paste an email to classify",
        "Congratulations! You've won a free gift card. Click here to claim your prize."
    )

    if st.button("Classify"):
        input_vectorized = vectorizer.transform([email_input])
        prediction = clf.predict(input_vectorized)
        prediction_proba = clf.predict_proba(input_vectorized)

        # Display results
        st.subheader("Prediction")
        st.write("**Spam**" if prediction[0] == 1 else "**Not Spam**")

        st.subheader("Prediction Probability")
        st.write(f"Spam: {prediction_proba[0][1]:.2f}, Not Spam: {prediction_proba[0][0]:.2f}")
