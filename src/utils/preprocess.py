import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import ast
import string

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

COOKING_STOPWORDS = {
    "fresh","freshly", "chopped", "optional", "pinch", "taste", "white","green","note","see","like",
    "small","large","medium","extra","pieces","piece", "adjust",
    "sliced","diced","minced","water","add","added","adding", "additional",
    "juice","frozen","half","red","handful","used","etc",
    "one","cut","shredded","peeled","use","thinly",
}

class Preprocessor:
    @staticmethod
    def clean_ingredients(ingredients_input):
        if isinstance(ingredients_input, list):
            ingredients_list = ingredients_input
        else:
            try:
                ingredients_list = ast.literal_eval(ingredients_input)
            except (ValueError, SyntaxError):
                return []

        cleaned_ingredients = []
        for ingredient in ingredients_list:
            cleaned = Preprocessor.clean_text(ingredient)
            if cleaned:
                cleaned_ingredients.append(cleaned)
        
        return list(set(cleaned_ingredients))

    @staticmethod
    def clean_text(text):  
        if pd.isna(text) or text is None or not isinstance(text, str):
            return "" # Return empty string instead of list for easier joining

        text = text.lower()
        # Remove numbers/fractions
        text = re.sub(r'\d+(\s\d+)?([/-]\d+)?', '', text) 
        
        unit_pattern = r'(tbl?s?(p(s)?)?\.?|tablespoons?|tsps?|teaspoons?|cups?|oz|ounces?|lbs?|pounds?|grams?|ml|quarts?|pints?|gallon|-inch|package|pkg\.|packets?|tube|sprigs?|t\.|-ish)\b\.?'
        text = re.sub(unit_pattern, '', text)
        text = text.strip()
        
        tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

        filtered_tokens = [word for word in lemmatized_tokens 
                        if word not in stop_words 
                        and word not in COOKING_STOPWORDS
                        and len(word) > 1 
                        and word not in string.punctuation
                        ]
        
        return ' '.join(filtered_tokens)
