import streamlit as st
import joblib

# Load the saved model and vectorizer
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit title
st.title("Twitter Sentiment Analysis")

# Input box for custom review
review = st.text_area("Enter your tweet:")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.error("Please enter a tweet to analyze.")
    else:
        try:
            # Transform the review using the saved TF-IDF vectorizer
            review_tfidf = vectorizer.transform([review])
            
            # Make prediction using the trained model
            prediction = model.predict(review_tfidf)
            
            # Convert prediction to a readable label
            prediction_label = 'positive' if prediction[0] == 1 else 'negative'
            
            # Get confidence level
            pred_proba = model.predict_proba(review_tfidf)
            confidence = pred_proba[0][1] if prediction[0] == 1 else pred_proba[0][0]

            # Display the result
            st.write(f"The sentiment of the review is: **{prediction_label}**")
            st.write(f"Confidence level: {confidence:.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
