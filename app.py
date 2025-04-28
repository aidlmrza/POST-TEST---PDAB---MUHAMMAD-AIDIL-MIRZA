from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Student GPA Prediction API")

# Load model
with open("XGBoost_Model.pkl", "rb") as f:
    model = pickle.load(f)

# Definisikan input data
class StudentData(BaseModel):
    Ethnicity: str
    ParentalEducation: str
    StudyTimeWeekly: float
    Absence: int
    Tutoring: str
    ParentalSupport: str
    Extracurricular: str
    Sports: str
    Music: str

# Fungsi untuk membuat fitur tambahan
def preprocess_input(data: StudentData):
    # Buat DataFrame dari input
    df = pd.DataFrame([{
        "Ethnicity": data.Ethnicity.lower(),
        "ParentalEducation": data.ParentalEducation.lower(),
        "StudyTimeWeekly": data.StudyTimeWeekly,
        "Absence": data.Absence,
        "Tutoring": data.Tutoring.lower(),
        "ParentalSupport": data.ParentalSupport.lower(),
        "Extracurricular": data.Extracurricular.lower(),
        "Sports": data.Sports.lower(),
        "Music": data.Music.lower(),
        "ParentalInvolvement": 0,
        "TotalActivities": 0
    }])

    # Encode ethnicity (misal: asian, black, hispanic, white)
    df['Ethnicity'] = df['Ethnicity'].map({
        'american': 0,
        'black american': 1,
        'asian': 2,
        'other': 3
    }).fillna(0).astype(int)
    
        # Encode parental_education (misal: highschool, bachelor, master)
    df['ParentalEducation'] = df['ParentalEducation'].map({
        'tidak ada': 0,
        'sma': 1,
        'kuliah': 2,
        'sarjana': 3,
        'magister': 4,
        'doktor': 4,
        'profesor':4
    }).fillna(0).astype(int)  # default ke 0 jika tidak ketemu

    df["Tutoring"] = df["Tutoring"].map({
        'yes': 1,
        'no': 0
    }).fillna(0).astype(int)

    df["ParentalSupport"] = df["ParentalSupport"].map({
        'tidak': 0,
        'rendah': 1,
        'sedang': 2,
        'tinggi': 3,
        'sangat tinggi': 4
    }).fillna(0).astype(int)

    df["Extracurricular"] = df["Extracurricular"].map({
        'iya': 1,
        'tidak': 0
    }).fillna(0).astype(int)

    df["Sports"] = df["Sports"].map({
        'iya': 1,
        'tidak': 0
    }).fillna(0).astype(int)

    df["Music"] = df["Music"].map({
        'iya': 1,
        'tidak': 0
    }).fillna(0).astype(int)

    # Buat kolom turunan
    df["ParentalInvolvement"] = (
        df["ParentalEducation"] + df["ParentalSupport"] +df["Tutoring"]
    )

    df["TotalActivities"] = (
        df["Extracurricular"] + df["Sports"] + df["Music"]
    )

    # Urutkan kolom supaya sesuai model
    df = df[[
        "Ethnicity",
        "ParentalEducation",
        "StudyTimeWeekly",
        "Absence",
        "Tutoring",
        "ParentalSupport",
        "Extracurricular",
        "Sports",
        "Music",
        "ParentalInvolvement",
        "TotalActivities"
    ]]

    return df

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "Student GPA Prediction API is running"}

# Endpoint untuk prediksi GPA
@app.post("/predict")
def predict_gpa(data: StudentData):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    
    return {
        "prediction_GPA": float(prediction)
    }

# Jalankan FastAPI dengan uvicorn
# uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)