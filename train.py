# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import joblib

# Download NLTK data (if not already downloaded)
try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Step 2: Load the Dataset
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]  # Keep only the label and message columns
    df.columns = ['label', 'message']  # Rename columns
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please provide the correct file path.")
    exit()

# Check class distribution
print("Class distribution:\n", df['label'].value_counts())

# Step 3: Preprocessing
# Convert labels to binary (spam = 1, ham = 0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing
df['message'] = df['message'].apply(preprocess_text)

# Step 4: Feature Extraction (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))  # Experiment with ngram_range
X = tfidf.fit_transform(df['message']).toarray()
y = df['label'].values

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)

# Metrics
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the Model
joblib.dump(model, 'spam_detection_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nModel and vectorizer saved successfully.")