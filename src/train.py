import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import os

from preprocessing import clean_text

# Download necessary NLTK data if not present
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def train_model():
    print("Loading dataset...")
    # Assuming the dataset is in the same directory as the script
    file_path = "../data/hate_speech_dataset.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        return

    df = pd.read_csv(file_path, encoding='utf-8')

    print("Preprocessing data...")
    # Standardize column names
    df = df.rename(columns={'Label': 'label', 'Content': 'text'})

    # Clean the labels and drop missing text
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df.dropna(subset=['text'], inplace=True)

    # Apply text cleaning
    df['clean_text'] = df['text'].apply(clean_text)
    print("Text Cleaning Done!")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42
    )

    print("Vectorizing text features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Saving model and vectorizer...")
    joblib.dump(model, '../models/hate_speech_model.pkl')
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    print("Optimization and saving complete! The model is ready for deployment.")

if __name__ == "__main__":
    train_model()
