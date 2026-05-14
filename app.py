import streamlit as st
import pickle
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = pickle.load(open('fake_job_model.pkl', 'rb'))

tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# NLP setup
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# Clean text function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Page config
st.set_page_config(
    page_title="Fake Job Predictor",
    page_icon="💼",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f5f5f5;
}

h1 {
    text-align: center;
    color: #333333;
}

.stButton>button {
    background-color: green;
    color: white;
    width: 100%;
    height: 50px;
    font-size: 20px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("Fake Job Predictor")

st.write(
    "Enter Job Description Details"
)

# Create 2 columns
col1, col2 = st.columns(2)

with col1:

    title = st.text_input("Title")

    department = st.text_input("Department")

    company_profile = st.text_area("Company Profile")

    requirements = st.text_area("Requirements")

    telecommuting = st.selectbox(
        "Telecommuting",
        ["Yes", "No"]
    )

    has_questions = st.selectbox(
        "Has Questions",
        ["Yes", "No"]
    )

    required_experience = st.text_input(
        "Required Experience"
    )

    industry = st.text_input("Industry")

with col2:

    location = st.text_input("Location")

    salary_range = st.text_input("Salary Range")

    description = st.text_area("Description")

    benefits = st.text_area("Benefits")

    has_company_logo = st.selectbox(
        "Has Company Logo",
        ["Yes", "No"]
    )

    employment_type = st.text_input(
        "Employment Type"
    )

    required_education = st.text_input(
        "Required Education"
    )

    function = st.text_input("Function")

# Submit button
if st.button("Submit"):

    # Combine all text
    full_text = (
        str(title) + " " +
        str(location) + " " +
        str(department) + " " +
        str(salary_range) + " " +
        str(company_profile) + " " +
        str(description) + " " +
        str(requirements) + " " +
        str(benefits) + " " +
        str(telecommuting) + " " +
        str(has_company_logo) + " " +
        str(has_questions) + " " +
        str(employment_type) + " " +
        str(required_experience) + " " +
        str(required_education) + " " +
        str(industry) + " " +
        str(function)
    )

    # Clean text
    cleaned_text = clean_text(full_text)

    # TF-IDF transform
    vector_input = tfidf.transform([cleaned_text])

    # Prediction
    prediction = model.predict(vector_input)

    # Result
    if prediction[0] == 1:

        st.error("🚨 Fake Job Posting")

    else:

        st.success("✅ Real Job Posting")