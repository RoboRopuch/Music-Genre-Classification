from typing import List, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np 
import datetime
import tensorflow as tf


app = FastAPI()

@app.on_event("startup")
def load_model():
    global le_genre, scaler, base_model, advanced_model, genres

    with open('../models/artifacts/encoders/le_genre.pkl', 'rb') as f:
        le_genre = pickle.load(f)
    with open('../models/artifacts/scalers/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    base_model = joblib.load('../models/artifacts/models/base_model.pkl')
    advanced_model = tf.keras.models.load_model('../models/artifacts/models/advanced_model.keras')

    genres = pd.read_csv('../data/genres.csv').columns[1:]


@app.get("/")
async def hello():
    print(genres)
    return {
        "message": "hello",
        }

@app.post("/base/prediction", response_model=str)
async def predict(input: List[float]) -> List[str]:
    data = await load_data(input)
    data = await preprocess_data(data)
    predictions = await predict_genre_base_model(data)
    predicted_genres = le_genre.inverse_transform(predictions)
    return predicted_genres[0]


@app.post("/advanced/prediction")
async def predict(input: List[float]) -> str:
    data = await load_data(input)
    data = await preprocess_data(data)
    predictions = await predict_genre_advanced_model(data)
    predicted_genres = [1 if pred >= 0.2 else 0 for pred in predictions[0]]
    predicted_genres_names = " ".join([genre*pred for genre, pred in zip(genres, predicted_genres)])
    return predicted_genres_names


@app.post("/compare/output", response_model=List[str])
async def compare_output(input: List[float]) -> List[str]:
    data = await load_data(input)
    data = await preprocess_data(data)
    base_model_prediction = await predict_genre_base_model(data)
    advanced_model_prediction = await predict_genre_advanced_model(data)
    decoded_prediction_base = le_genre.inverse_transform(base_model_prediction)
    decoded_prediction_advanced = le_genre.inverse_transform(advanced_model_prediction)
    response_data = {
        "base_model": decoded_prediction_base.tolist(),
        "advanced_model": decoded_prediction_advanced.tolist()
    }
    return JSONResponse(content=response_data)

@app.get("/compare/accuracy")
async def compare_accuracy():
    base_accuracy = await calculate_accuracy_base()
    response_data = {
        "base_model": base_accuracy,
        "advanced_model": 'laoding'
    }
    return JSONResponse(content=response_data)

@app.get("/ABexperiment")
async def perform_experiment():
    session_df = pd.read_json("../data/v2/sessions.jsonl", lines=True)
    session_summary_df = await generate_sessions_summary(session_df)
    group_A = pd.read_csv('../data/experiment/groupA.csv') 
    group_B = pd.read_csv('../data/experiment/groupB.csv') 

    mean_duration_A = calculate_mean_duration_group(group_A,session_summary_df)
    mean_duration_B = calculate_mean_duration_group(group_B, session_summary_df)

    response_data = {
        "no model": mean_duration_A,
        "model": mean_duration_B,
        "time": datetime.datetime.now().isoformat()
    }

    with open("../data/experiment/experiment_history.jsonl", "a") as f:
        f.write(json.dumps(response_data) + '\n')

    return JSONResponse(content=response_data)


#ACCURACY CALCULATION HELPERS
def load_test_data_base():
    test_data_base_df = pd.read_csv('../data/accuracy/base.csv')
    return test_data_base_df

def calculate_accuracy_base():
    test_data_base_df = load_test_data_base()
    X = test_data_base_df.drop(columns=['id', 'genre', 'genre_encoded'])
    y = test_data_base_df['genre_encoded']
    #czy dzielić na zbiór testowy?
    base_predictions = base_model.predict(X)
    return accuracy_score(y, base_predictions)

#AB EXPERIMENT HELPERS
async def generate_sessions_summary(session_df: pd.DataFrame) -> List[int]:
    #calculate duration of each session
    session_df['timestamp'] = pd.to_datetime(session_df['timestamp'])
    session_df = session_df.sort_values(by='timestamp')
    session_df['duration'] = session_df.groupby('session_id')['timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())

    #calculate interactions for each session
    session_summary_df = session_df.groupby('session_id').agg(
        duration=('duration', 'first'),
        user_id=('user_id', 'first'),
        num_play_events=('event_type', lambda x: (x == 'play').sum()),
        num_like_events=('event_type', lambda x: (x == 'like').sum()),
        num_skip_events=('event_type', lambda x: (x == 'skip').sum())
    ).reset_index()

    session_summary_df = session_summary_df.set_index('session_id')
    return session_summary_df

def calculate_mean_duration_user(user, session_summary_df):
    user_id = user['user_id']
    user_sessions_df = session_summary_df.loc[session_summary_df['user_id'] == user_id].copy()
    return np.mean(user_sessions_df['duration'])
    
def calculate_mean_duration_group(group, session_summary_df):
    users_means = []
    for index, row in group.iterrows():
        mean_session_duration = calculate_mean_duration_user(row.copy(), session_summary_df)
        users_means.append(mean_session_duration)
    return np.mean(users_means)


#PREDICTION HELPERS
async def predict_genre_base_model(data: pd.DataFrame) -> List[int]:
    prediction = base_model.predict(data)
    return prediction

async def predict_genre_advanced_model(data: pd.DataFrame) -> List[int]:
    prediction = advanced_model.predict(data)
    return prediction

async def load_data(data: List[List[Any]]) -> pd.DataFrame:
    print(data)
    labels = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 
              'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
              'time_signature', 'release_year']
    df = pd.DataFrame([data], columns=labels)
    return df

async def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    features_to_normalize = ['popularity', 'duration_ms', 'danceability', 'key', 'loudness', 'speechiness', 
                             'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                             'release_year', 'time_signature']
    processed_data = data.copy()
    processed_data[features_to_normalize] = scaler.transform(data[features_to_normalize])
    return processed_data
