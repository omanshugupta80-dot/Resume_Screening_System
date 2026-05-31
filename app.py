from flask import Flask, render_template, request
import os
import joblib

from utils.parser import extract_resume_text
from utils.preprocess import clean_resume
from utils.skills import extract_skills

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load saved files
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "resume" not in request.files:
        return "No file uploaded"

    file = request.files["resume"]

    if file.filename == "":
        return "Please select a file"

    # File validation
    allowed_extensions = [".pdf", ".docx"]

    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        return "Only PDF and DOCX files are allowed"

    # Save file
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    # Extract text
    resume_text = extract_resume_text(file_path)

    if not resume_text:
        return "Unable to extract text from resume"

    # Clean text
    cleaned_text = clean_resume(resume_text)

    # Vectorize
    vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vector)

    predicted_role = label_encoder.inverse_transform(prediction)[0]

    # Confidence score
    confidence = max(model.predict_proba(vector)[0]) * 100

    # Skill extraction
    skills = extract_skills(resume_text)

    return render_template(
        "result.html",
        category=predicted_role,
        confidence=round(confidence, 2),
        skills=skills
    )


if __name__ == "__main__":
    app.run(debug=True)
    