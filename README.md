# 🎬 Movie Success Prediction using XGBoost & SHAP

## 📌 Overview
This project predicts whether a movie will be **successful** based on IMDb metadata such as runtime, ratings, number of votes, and genres.  
It is an **end-to-end data science project** that includes data preprocessing, feature engineering, model training, explainability, and deployment using Streamlit.

---

## 🎯 Objective
To build a machine learning system that:
- Works on real-world IMDb data
- Predicts movie success accurately
- Explains predictions using interpretable ML techniques
- Provides an interactive web interface

---

## 📊 Dataset
IMDb public datasets in **TSV format**:

- `title.basics.tsv` – movie metadata (runtime, genres, title type)
- `title.ratings.tsv` – IMDb ratings and vote counts

The original dataset contains millions of records.  
A random sample is used during training for performance reasons.

---

## 🧠 Feature Engineering

### Numerical Features
- Runtime (minutes)
- Average IMDb rating
- Number of votes

### Genre Features (One-Hot Encoded)
- Action
- Comedy
- Drama
- Romance
- Thriller
- Horror
- Adventure
- Crime
- Sci-Fi
- Fantasy

### Target Variable
A movie is labeled as **successful (1)** if:
- Average rating ≥ 7  
- Number of votes ≥ 1000  

Otherwise, it is labeled as **not successful (0)**.

---

## 🤖 Model
### XGBoost Classifier
XGBoost was chosen because:
- It captures non-linear relationships
- Performs well on imbalanced datasets
- Is widely used in industry
- Provides feature importance scores

### Class Imbalance Handling
The `scale_pos_weight` parameter is used to balance successful and unsuccessful movies.

---

## 🔍 Explainability with SHAP
SHAP (SHapley Additive exPlanations) is used to:
- Explain individual predictions
- Show how each feature contributes to success or failure
- Provide transparent and interpretable results

Both **local explanations** (single prediction) and **global feature importance** are shown in the app.

---

## 🌐 Web Application
The Streamlit web app allows users to:
- Adjust runtime, rating, and vote count
- Select multiple genres
- View success probability
- Understand predictions using SHAP values
- Explore overall feature importance

---

## 📁 Project Structure
movies/
│
├── main.py # Model training (XGBoost)
├── app.py # Streamlit web application
├── movie_success_model.pkl # Trained model
├── title.basics.tsv # IMDb dataset
├── title.ratings.tsv # IMDb dataset
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

▶️ How to Run

1️⃣ Install Dependencies

    Ensure Python 3.9 or higher is installed.
    Install all required libraries using:

    pip install -r requirements.txt

2️⃣ Train the Model

    Run the training script:
    python main.py

    This generates the trained model file:
    movie_success_model.pkl

3️⃣ Run the Web Application

    Start the Streamlit app:
    streamlit run app.py

    The application opens at:
    http://localhost:8501

4️⃣ Use the Application

- Adjust runtime, rating, and number of votes
- Select one or more genres
- Click Predict Success
- View success probability and SHAP explanations