import pandas as pd
from src.utils.preprocess import clean_ingredients
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os
import pickle

load_dotenv()
RAW_PATH = os.getenv("RAW_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
TFIDF_MATRIX_PATH = os.getenv("TFIDF_MATRIX_PATH")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH")
RECIPES_PATH = os.getenv("RECIPES_PATH")

df = pd.read_csv(RAW_PATH)

df['ingredients_cleaned'] = df['ingredients'].apply(clean_ingredients)
df['ingredients_cleaned'] = df['ingredients_cleaned'].apply(lambda x: ' '.join(x))
df.to_csv(OUTPUT_PATH, index=False)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=4,  # Ignore terms in <4 docs (reduces noise)
    max_df=0.8,  # Ignore terms in >80% of docs (removes overly common)
)
tfidf_matrix = vectorizer.fit_transform(df['ingredients_cleaned'])

features = vectorizer.get_feature_names_out()
bigrams = [f for f in features if " " in f]
print(bigrams[:50])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

with open(TFIDF_MATRIX_PATH, 'wb') as file:
    pickle.dump(tfidf_matrix, file)

with open(VECTORIZER_PATH, 'wb') as file:
    pickle.dump(vectorizer, file)

with open(RECIPES_PATH, 'wb') as file:
    pickle.dump(df, file)
print("TF-IDF matrix and vectorizer saved to 'tfidf_matrix.pkl' and 'tfidf_vectorizer.pkl'.")

