from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import clean_text

# ---------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Hate Speech Detection System",
    page_icon="🛑",
    layout="centered"
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "hate_speech_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_artifacts()

# ---------------------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------------------

def predict_text(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = probabilities[prediction]

    return prediction, confidence, probabilities


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("🛑 Hate Speech Detection")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Project Overview")

st.sidebar.write(
    "Predict whether a sentence contains hate speech using Natural Language Processing and Machine Learning."
)

st.sidebar.markdown("---")

st.sidebar.subheader("🧠 Model")

st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Vectorizer:** TF-IDF")
st.sidebar.write("**Task:** Binary Classification")

st.sidebar.markdown("---")

st.sidebar.subheader("🛠 Technologies")

st.sidebar.markdown("""
- Python
- Scikit-Learn
- Streamlit
- NLTK
- Pandas
""")

# ---------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------

st.title("🛑 Hate Speech Detection System")

st.write(
    "Analyze text using a Machine Learning model trained with "
    "**TF-IDF** features and **Logistic Regression**."
)

st.info(
    "This application demonstrates a Machine Learning pipeline for hate speech detection using TF-IDF vectorization and Logistic Regression."
)

st.markdown("---")

st.subheader("📝 Enter Text")

examples = {
    "😊 Positive": "I hope you have a wonderful day and achieve all your goals.",
    "😐 Neutral": "you need to buckle up now",
    "⚠️ Hate Speech": "You are an idiot and nobody likes you."
}

selected_example = st.selectbox(
    "Try an Example",
    ["None"] + list(examples.keys())
)

default_text = ""

if selected_example != "None":
    default_text = examples[selected_example]

user_text = st.text_area(
    "Text",
    value=default_text,
    height=170,
    placeholder="Type or paste text here..."
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Characters", len(user_text))

with col2:
    st.metric("Words", len(user_text.split()))

st.caption(
    "Enter a sentence manually or choose one of the sample inputs above."
)

analyze = st.button(
    "Analyze Text",
    type="primary",
    use_container_width=True
)

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

if analyze:

    if not user_text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing text..."):
        prediction, confidence, probabilities = predict_text(user_text)

    st.markdown("---")

    if prediction == 1:
        st.error("### ⚠️ Hate Speech Detected")
    else:
        st.success("### ✅ Non Hate Speech")

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.progress(float(confidence))

    if confidence >= 0.80:
        st.success("Reliability: High Confidence")
    elif confidence >= 0.60:
        st.warning("Reliability: Moderate Confidence")
    else:
        st.info("Reliability: Low Confidence")

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Class": [
            "Non Hate Speech",
            "Hate Speech"
        ],
        "Probability": [
            f"{probabilities[0]*100:.2f}%",
            f"{probabilities[1]*100:.2f}%"
        ]
    })

    st.table(probability_df)

    st.subheader("Probability Distribution")

    chart_df = pd.DataFrame(
        {
            "Probability": [
                probabilities[0],
                probabilities[1]
            ]
        },
        index=[
            "Non Hate Speech",
            "Hate Speech"
        ]
    )

    st.bar_chart(chart_df)

# ---------------------------------------------------------------------
# About Model
# ---------------------------------------------------------------------

with st.expander("ℹ About the Project"):

    st.markdown("""
### Workflow

1. Text Preprocessing
2. TF-IDF Feature Extraction
3. Logistic Regression Prediction
4. Confidence Estimation

---

### Model Information

- Logistic Regression
- TF-IDF Vectorization
- Binary Text Classification
- Scikit-Learn Pipeline

---

### Technologies

- Python
- Streamlit
- Scikit-Learn
- NLTK
- Pandas

---

### Limitations

- Intended for educational purposes.
- May misclassify sarcasm or context-dependent language.
- Performance depends on the quality and diversity of the training dataset.
""")
    
    
# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.markdown("---")

st.markdown("""Built using Python, Scikit-Learn, and Streamlit""")

st.markdown("""
### 👤 Developer
**Charagundla Hemanth Sai**
""")


st.markdown("""
### Contact Information

🎯 **Interested in:** Data Analytics • Business Intelligence • Data Engineering • Machine Learning

📧 **Email:** hemanthsai.ch456@gmail.com

🔗 **LinkedIn:** https://www.linkedin.com/in/hemanthsai456/

💻 **GitHub:** https://github.com/Hemanthsai456

💻 **GitHub Repository:** https://github.com/Hemanthsai456/Hate-Speech-Detection-System
""")