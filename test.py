import numpy as np
import streamlit as st
import pickle

# Load the multi-output model and vectorizer
with open('multi_target_model.pkl', 'rb') as f:
    multi_target_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title('Email Classifier')

user_input = st.text_input('Enter your email:')
if st.button('Classify'):
    input_data = [user_input]
    input_vectorized = vectorizer.transform(input_data)

    prediction_probs = []
    for estimator in multi_target_model.estimators_:
        decision = estimator.decision_function(input_vectorized)
        proba = 1 / (1 + np.exp(-decision))  # Sigmoid function
        prediction_probs.append(proba)

    threshold = 0.5  # Adjust the threshold as needed

    # Check if the email is not an advertisement
    if prediction_probs[0][0] <= threshold:
        st.write("Not an advertisement. No other classifications available.")
    else:
        # Find the index of the maximum probability for age groups
        max_age_index = np.argmax(prediction_probs[2:])

        st.write('Advertisement:', 'Yes' if prediction_probs[0][0] > threshold else 'No')
        st.write('Spam:', 'Yes' if prediction_probs[1][0] > threshold else 'No')
        st.write('Children:', 'Yes' if max_age_index == 0 else 'No')
        st.write('Young Adult:', 'Yes' if max_age_index == 1 else 'No')
        st.write('Adult:', 'Yes' if max_age_index == 2 else 'No')
