import streamlit as st
from src.recommender import RecipeRecommender
import pickle

@st.cache_resource
def load_recommender():
    return RecipeRecommender()

@st.cache_resource
def load_calorie_assets():
    with open("src/models/calories_regressor.pkl", "rb") as f:
        model = pickle.load(f)

    with open("src/models/tfidf_vectorizer_calories.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("src/models/calorie_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("src/models/was_log_transformed.pkl", "rb") as f:
        was_log_transformed = pickle.load(f)

    with open("src/models/calorie_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)

    return {
        "regressor": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "was_log_transformed": was_log_transformed,
        "classifier": classifier,
    }

def ensure_models_loaded():
    try:
        if "recommender" not in st.session_state:
            with st.spinner("Loading recipe recommender..."):
                st.session_state.recommender = load_recommender()

        if "calorie_models" not in st.session_state:
            with st.spinner("Loading calorie models..."):
                st.session_state.calorie_models = load_calorie_assets()

    except Exception as e:
        st.error("Failed to load required models.")
        st.exception(e)
        st.stop()
