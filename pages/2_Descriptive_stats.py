import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

st.set_page_config(page_title="Recipe EDA", layout="wide")

st.title("Recipe Descriptive Statistics & EDA")

# ------------------------
# Load data
# ------------------------
@st.cache_data
def load_data():
    with open("data/processed/recipes.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()

# ------------------------
# Feature Engineering
# ------------------------
df["num_ingredients"] = df["ingredients"].apply(lambda x: len(x.split()))
df["calories_per_ingredient"] = df["calories"] / df["num_ingredients"]

def count_steps(instructions):
    if isinstance(instructions, list):
        return len(instructions)
    return instructions.count(".")

df["num_steps"] = df["instructions"].apply(count_steps)

def difficulty(steps):
    if steps <= 5:
        return "Easy"
    elif steps <= 10:
        return "Medium"
    return "Hard"

df["difficulty"] = df["num_steps"].apply(difficulty)

MEAT = ["chicken", "beef", "pork", "lamb", "bacon", "turkey"]
df["is_vegetarian"] = df["ingredients"].apply(
    lambda x: not any(m in x for m in MEAT)
)

HIGH_CAL = ["oil", "butter", "olive"]
df["has_oil_butter"] = df["ingredients"].apply(
    lambda x: any(h in x for h in HIGH_CAL)
)

# ------------------------
# 1. Calories Distribution
# ------------------------
st.header("1️- Distribution of Calories")

fig, ax = plt.subplots()
sns.histplot(df["calories"], bins=40, kde=True, ax=ax)
st.pyplot(fig)

# ------------------------
# 2. Difficulty vs Steps
# ------------------------
st.header("2- Recipe Difficulty vs Steps")

fig, ax = plt.subplots()
sns.boxplot(data=df, x="difficulty", y="num_steps", ax=ax)
st.pyplot(fig)

# ------------------------
# 3. Ingredient Frequency
# ------------------------
st.header("3- Ingredient Frequency Distribution")

all_ingredients = " ".join(df["ingredients_cleaned"]).split()
ingredient_counts = Counter(all_ingredients)

ingredient_freq_df = (
    pd.DataFrame(ingredient_counts.items(), columns=["ingredient", "count"])
    .sort_values(by="count", ascending=False)
    .head(20)
)

fig, ax = plt.subplots()
sns.barplot(data=ingredient_freq_df, y="ingredient", x="count", ax=ax)
st.pyplot(fig)

# ------------------------
# 4-. Oil / Butter Impact
# ------------------------
st.header("4- Calories vs Oil / Butter")

fig, ax = plt.subplots()
sns.boxplot(data=df, x="has_oil_butter", y="calories", ax=ax)
st.pyplot(fig)

# ------------------------
# 5. Vegetarian vs Non-Vegetarian
# ------------------------
st.header("5- Vegetarian vs Non-Vegetarian Calories")

fig, ax = plt.subplots()
sns.boxplot(data=df, x="is_vegetarian", y="calories", ax=ax)
st.pyplot(fig)

st.success("EDA dashboard loaded successfully")