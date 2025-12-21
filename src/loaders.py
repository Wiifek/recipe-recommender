import streamlit as st
import pickle
from src.recommender import RecipeRecommender

class ModelLoader:
    # Responsible for loading and caching of all machine learning models and assets

    @staticmethod
    @st.cache_resource
    def load_recommender():
        # Loads the main recipe recommender
        return RecipeRecommender()

    @staticmethod
    @st.cache_resource
    def load_calorie_assets():
        # Loads all models related to calorie prediction
        paths = {
            "regressor": "src/models/calories_regressor.pkl",
            "vectorizer": "src/models/tfidf_vectorizer_calories.pkl",
            "scaler": "src/models/calorie_scaler.pkl",
            "was_log_transformed": "src/models/was_log_transformed.pkl",
            "classifier": "src/models/calorie_classifier.pkl"
        }
        
        assets = {}
        for key, path in paths.items():
            with open(path, "rb") as f:
                assets[key] = pickle.load(f)
        return assets

    @classmethod
    def ensure_models_loaded(cls):
        try:
            if "recommender" not in st.session_state:
                with st.spinner("Loading recipe recommender..."):
                    st.session_state.recommender = cls.load_recommender()

            if "calorie_models" not in st.session_state:
                with st.spinner("Loading calorie models..."):
                    st.session_state.calorie_models = cls.load_calorie_assets()

        except Exception as e:
            st.error("Failed to load required models.")
            st.exception(e)
            st.stop()
