import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from utils.preprocess import clean_resume

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/UpdatedResumeDataSet.csv")

# Check column names
print(df.columns)

# Clean resumes
df["cleaned_resume"] = df["Resume"].apply(clean_resume)

# Features and labels
X = df["cleaned_resume"]
y = df["Category"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Train model
model = SVC(kernel="linear", probability=True)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy*100:.2f}%")

# Save files
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/tfidf.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("Model Saved Successfully")