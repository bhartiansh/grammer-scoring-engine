import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

MODEL_PATH = "model/grammar_model.joblib"

def train_model(transcript_csv_path, train_csv_path):
    transcripts = pd.read_csv(transcript_csv_path)
    labels = pd.read_csv(train_csv_path)

    merged = pd.merge(labels, transcripts, on="filename", how="inner")
    merged = merged.dropna(subset=["transcript", "label"])

    X = merged["transcript"]
    y = merged["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LinearRegression())
    ])
    pipeline.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

def predict_scores(transcripts):
    if isinstance(transcripts, pd.Series):
        transcripts = transcripts.tolist()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found.")
    pipeline = joblib.load(MODEL_PATH)
    preds = pipeline.predict(transcripts)
    return np.clip(preds, 0, 5)
