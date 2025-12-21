import pandas as pd
import numpy as np
import os
import pickle
import warnings
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

warnings.filterwarnings("ignore")

class CaloriesRegression:
    def __init__(self):
        load_dotenv()
        self.data_path = os.getenv("OUTPUT_PATH", "../data/processed/recipes_clean.csv")
        self.model_dir = "src/models/"
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_df=0.8)
        self.scaler = StandardScaler(with_mean=False)
        self.regressor = None
        self.classifier = None
        self.was_log_transformed = False
        
        self.df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.data_path)
        y = self.df['calories']
        
        # Log transformation logic
        self.was_log_transformed = np.abs(y.skew()) > 1
        y_processed = np.log1p(y) if self.was_log_transformed else y.copy()
            
        # Fitting the calorie_vectorizer
        X_tfidf = self.calorie_vectorizer.fit_transform(self.df['ingredients_cleaned'])
        
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X_tfidf, y_processed, test_size=0.2, random_state=42
        )
        
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

    def train_regressor(self):
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': [None, 10, 20],
            'min_samples_split': randint(2, 10)
        }
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            param_distributions=param_dist,
            n_iter=20, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
        )
        search.fit(self.X_train_scaled, self.y_train)
        self.regressor = search.best_estimator_

    def train_classifier(self):
        def classify(cps):
            if cps < 400: return 'Low'
            elif cps <= 800: return 'Medium'
            else: return 'High'

        self.df['calorie_class'] = self.df['calories'].apply(classify)
        X_class = self.calorie_vectorizer.transform(self.df['ingredients_cleaned'])
        y_class = self.df['calorie_class']

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
        )

        self.classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.classifier.fit(X_train_c, y_train_c)

    def save_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        artifacts = {
            "calories_regressor.pkl": self.regressor,
            "tfidf_vectorizer_calories.pkl": self.vectorizer,
            "calorie_scaler.pkl": self.scaler,
            "was_log_transformed.pkl": self.was_log_transformed,
            "calorie_classifier.pkl": self.classifier
        }
        for filename, obj in artifacts.items():
            with open(os.path.join(self.model_dir, filename), "wb") as f:
                pickle.dump(obj, f)

    def run_full_pipeline(self):
        self.load_and_prepare_data()
        self.train_regressor()
        self.train_classifier()
        self.save_models()

if __name__ == "__main__":
    trainer = CaloriesRegression()
    trainer.run_full_pipeline()