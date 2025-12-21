import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

VECTORIZER_PATH = os.getenv("VECTORIZER_PATH")
TFIDF_MATRIX_PATH = os.getenv("TFIDF_MATRIX_PATH")
RECIPES_PATH = os.getenv("RECIPES_PATH")

class RecipeRecommender:
    def __init__(self):
        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(TFIDF_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        with open(RECIPES_PATH, "rb") as f:
            self.df = pickle.load(f)

        self.df = self.df.reset_index(drop=True).copy()

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend_by_ingredients(
        self,
        input_ingredients,
        max_calories=None,
        top_k=5
    )-> pd.DataFrame:
        if not input_ingredients:
            raise ValueError("Input ingredients cannot be empty.")
        
        input_text = " ".join(input_ingredients)
        input_vector = self.vectorizer.transform([input_text])

        similarity_scores = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
        
        filtered_df = self.df.copy()
        filtered_df["similarity"] = similarity_scores

        if max_calories is not None:
            filtered_df = filtered_df[filtered_df["calories"] <= max_calories]

        recommendations = (
            filtered_df.sort_values(by="similarity", ascending=False)
                  .head(top_k)
                  .copy()
        )
        cols = ["image", "title", "calories", "serves", "total time", "similarity", "ingredients"]
        available_cols = [col for col in cols if col in recommendations.columns]
        return recommendations[available_cols]
    
    def get_most_used_ingredients(self, top_n: int = 20) -> pd.DataFrame:
        all_ingredients = " ".join(self.df["ingredients_cleaned"].astype(str)).split()
        counter = Counter(all_ingredients)
        return pd.DataFrame(counter.most_common(top_n), columns=["ingredient_cleaned", "count"])
