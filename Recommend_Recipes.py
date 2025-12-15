import streamlit as st
import ast
from src.loaders import ensure_models_loaded

ensure_models_loaded()
recommender = st.session_state.recommender

st.header("Find Recipes by Ingredients")

# Ingredient input
user_input = st.text_input(
    "Enter ingredients you have (comma-separated):",
    placeholder="e.g. chicken, rice, tomato, garlic"
)

col1, col2 = st.columns(2)
with col1:
    max_calories = st.slider(
        "Max calories per serving",
        min_value=0,
        max_value=4000,
        value=2000,
        step=100
    )

with col2:
    top_k = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

if st.button("Find Recipes"):

    if not user_input.strip():
        st.warning("Please enter at least one ingredient.")
        st.stop()

    ingredients = [
        ing.strip().lower()
        for ing in user_input.split(",")
        if ing.strip()
    ]

    with st.spinner("Searching for matching recipes..."):
        results = recommender.recommend_by_ingredients(
            input_ingredients=ingredients,
            top_k=top_k,
            max_calories=max_calories
        )

    # ---------- RESULTS ----------
    if results.empty:
        st.warning("No recipes found! Try fewer or broader ingredients.")
    else:
        st.success(f"Found {len(results)} great matches!")

        for _, row in results.iterrows():
            with st.expander(
                f"{row['title']} • {row['calories']} cal • "
                f"Similarity: {row['similarity']:.2f}"
            ):
                if row.get("image"):
                    st.image(row["image"], width=250)

                st.markdown("### 🧂 Ingredients")
                orig_list = ast.literal_eval(row["ingredients"])
                for item in orig_list:
                    st.write(f"• {item}")

                if "serves" in row and row["serves"]:
                    st.write(f"**Serves:** {row['serves']}")

                if "total time" in row and row["total time"]:
                    st.write(f"**Time:** {row['total time']}")

else:
    st.info("Enter ingredients and click **Find Recipes** to begin")