import pandas as pd
import os
import pickle
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.preprocess import Preprocessor

class RecipeVectorizer:
    def __init__(self):
        """Initialize paths and configuration from environment variables."""
        load_dotenv()
        self.raw_path = os.getenv("RAW_PATH")
        self.output_path = os.getenv("OUTPUT_PATH")
        self.tfidf_matrix_path = os.getenv("TFIDF_MATRIX_PATH")
        self.vectorizer_path = os.getenv("VECTORIZER_PATH")
        self.recipes_path = os.getenv("RECIPES_PATH")
        
        self.df = None
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=4,
            max_df=0.8
        )
        self.tfidf_matrix = None

    def load_and_preprocess(self):
        """Loads raw data and applies ingredient cleaning."""
        print(f"Loading data from {self.raw_path}...")
        self.df = pd.read_csv(self.raw_path)

        print("Cleaning ingredients...")
        self.df['ingredients_cleaned'] = self.df['ingredients'].apply(Preprocessor.clean_ingredients)
        self.df['ingredients_cleaned'] = self.df['ingredients_cleaned'].apply(lambda x: ' '.join(x))
        
        # Save cleaned CSV
        self.df.to_csv(self.output_path, index=False)
        print(f"Cleaned data saved to {self.output_path}")

    def train_vectorizer(self):
        """Fits the TF-IDF vectorizer to the cleaned ingredients."""
        if self.df is None:
            raise ValueError("Dataframe is empty. Run load_and_preprocess() first.")

        print("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['ingredients_cleaned'])
        
        # Log a sample of bigrams for verification
        features = self.vectorizer.get_feature_names_out()
        bigrams = [f for f in features if " " in f]
        print(f"Top 50 Bigrams: {bigrams[:50]}")

    def save_models(self):
        artifacts = {
            self.tfidf_matrix_path: self.tfidf_matrix,
            self.vectorizer_path: self.vectorizer,
            self.recipes_path: self.df
        }

        for path, obj in artifacts.items():
            with open(path, 'wb') as file:
                pickle.dump(obj, file)
        
        print(f"All models saved to designated paths.")

    def run_pipeline(self):
        self.load_and_preprocess()
        self.train_vectorizer()
        self.save_models()

if __name__ == "__main__":
    trainer = RecipeVectorizer()
    trainer.run_pipeline()
