import streamlit as st
import numpy as np
import pandas as pd
from src.utils.preprocess import clean_ingredients

calorie_models = st.session_state.calorie_models

st.header("🔮 Predict Calories from Ingredients")

# User input
user_input = st.text_area(
    "Enter ingredients (one per line or comma-separated):",
)

if st.button("Predict Calories"):
    if user_input:
        # Split and clean input
        ingredients_list = [ing.strip() for ing in user_input.replace(",", "\n").split("\n") if ing.strip()]
        cleaned_ingredients = clean_ingredients(str(ingredients_list))
        cleaned_text = " ".join(cleaned_ingredients)
        
        if not cleaned_text:
            st.warning("No valid ingredients after cleaning. Try again!")
        else:
            # Vectorize and predict
            X_new = calorie_models["vectorizer"].transform([cleaned_text])
            X_new_dense = calorie_models["scaler"].transform(X_new)

            #Regression
            pred = calorie_models["regressor"].predict(X_new_dense)[0]
            #classification
            class_pred = calorie_models["classifier"].predict(X_new)[0]
            
            if calorie_models["was_log_transformed"]:
                pred = np.expm1(pred)
            calories_per_serving = round(pred)
            
            st.success(f"Predicted Calories: **{pred:.0f}** (per serving) - **{class_pred} calories** recipe")
    else:
        st.info("Enter some ingredients to predict!")
