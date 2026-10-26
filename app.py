import streamlit as st
import joblib


#Loading-model

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


#Page-Configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)


##Title

st.title("📧 Email Spam Detector")

st.write(
    "Enter an email subject and body below. "
    "The machine learning model will classify "
    "the email as Spam or Not Spam."
)


#Input

subject = st.text_input(
    "Email Subject",
    placeholder="Enter email subject..."
)

body = st.text_area(
    "Email Body",
    placeholder="Enter email content...",
    height=220
)


#Prediction

if st.button("Check Email"):

    if not subject.strip() and not body.strip():

        st.warning("Please enter an email.")

    else:

        # Combine subject and body
        email_text = subject + " " + body

        # Convert email to TF-IDF
        email_vector = vectorizer.transform(
            [email_text]
        )

        # Make prediction
        prediction = model.predict(
            email_vector
        )[0]

        # Get probabilities
        probabilities = model.predict_proba(
            email_vector
        )[0]

        spam_probability = probabilities[1]
        not_spam_probability = probabilities[0]


# Display Result

        if prediction == 1:

            st.error("⚠️ SPAM EMAIL")

            st.metric(
                "Spam Probability",
                f"{spam_probability * 100:.2f}%"
            )

        else:

            st.success("✓ NOT SPAM")

            st.metric(
                "Not Spam Probability",
                f"{not_spam_probability * 100:.2f}%"
            )