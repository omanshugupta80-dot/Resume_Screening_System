SKILLS_DB = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "html",
    "css",
    "javascript",
    "flask",
    "django",
    "react",
    "nodejs",
    "mongodb",
    "mysql",
    "data science",
    "pandas",
    "numpy",
    "communication",
    "leadership"
]



def extract_skills(text):
    text = text.lower()

    found_skills = []

    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)

    return found_skills