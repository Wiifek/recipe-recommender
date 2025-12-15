import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from dotenv import load_dotenv
import os
load_dotenv()
RECIPE_LIST_URL = os.getenv("RECIPE_LIST_URL")
HEADERS = {
    "User-Agent": os.getenv("USER_AGENT")
}

def get_number_of_pages():
    response = requests.get(RECIPE_LIST_URL, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    dots_span = soup.find('span', class_='page-numbers dots')
    if dots_span:
        next_page_link = dots_span.find_next('a', class_='page-numbers')
        if next_page_link:
            total_pages = int(next_page_link.get_text(strip=True))
            return total_pages

    return 1

def get_recipe_links(total_pages):
    recipe_links = []
    for page in range(1, total_pages+1):
        page_url = f"{RECIPE_LIST_URL}/page/{page}"
        page_response = requests.get(page_url, headers=HEADERS)
        if page_response.status_code != 200:
            print(f"Failed to fetch page {page}")
            continue

        page_soup = BeautifulSoup(page_response.content, "html.parser")
        articles_container = page_soup.find("div", class_="grid grid-cols-12 gap-4")
        articles = articles_container.find_all("article")
        for article in articles:
             # append link only if title does NOT start with a digit
            if not re.search(r'^\d',article.find("h3").get_text()): 
                recipe_links.append(article.find("a")["href"])
    return recipe_links


def get_recipes(recipe_url):
    try:
        recipe_response = requests.get(recipe_url, headers=HEADERS)
        recipe_soup = BeautifulSoup(recipe_response.content, "html.parser")
        recipe_div = recipe_soup.find("div", class_="tasty-recipes")
        recipe_img = recipe_div.find("img", class_="attachment-featured-medium size-featured-medium")["src"]
        recipe_title = recipe_div.find("h2", class_="tasty-recipes-title").text
        recipe_total_time = recipe_div.find("span", class_="tasty-recipes-total-time").text

        try:
            recipe_description = recipe_div.find("div", class_= "tasty-recipes-description-body").find("p").text
        except Exception:
            recipe_description = ""
        recipe_ingredients_header = recipe_div.find("div", class_= "tasty-recipes-ingredients-header")
        recipe_ingredients = recipe_ingredients_header.find_next_sibling("div")
        ingredients = []
        for li_tag in recipe_ingredients.findAll('li'):
            ingredients.append(li_tag.input["aria-label"])

        instructions = []
        recipe_instructions_header = recipe_div.find("div", class_="tasty-recipes-instructions-header")
        recipe_instructions = recipe_instructions_header.find_next_sibling("div")
        instruction_items = recipe_instructions.findAll("li")
        for item in instruction_items:
            instructions.append(item.text)
        
        nutrition_facts_div = recipe_div.find("div", class_="tasty-recipes-nutrifox")
        nutrition_facts_link = nutrition_facts_div.find("iframe")["src"]
        serves, calories_per_serving = get_recipe_nutrition(nutrition_facts_link)

        recipe = {'image': recipe_img,
                  'title': recipe_title,
                  'description': recipe_description,
                  'total time': recipe_total_time,
                  'ingredients': ingredients,
                  'instructions': instructions,
                  'calories': calories_per_serving,
                  'serves': serves
                  }
        return recipe
    except Exception as e:
        print(f"Exception {e} occurred while scraping {recipe_url}")
        return None

def get_recipe_nutrition(nutrition_facts_link):
    try:
        nutrition_response = requests.get(nutrition_facts_link, headers=HEADERS)
        nutrition_soup = BeautifulSoup(nutrition_response.content, "html.parser")
        script_tag = nutrition_soup.find("script", string=lambda s: s and "var preloaded" in s)
        script_text = script_tag.string
        servings_match = re.search(r"\"servings\":\d+", script_text)
        calories_match = re.search(r"\"calories\":\d+.\d+", script_text)
        if servings_match and calories_match:
            servings = int(servings_match.group().split(":")[1])
            calories = int(round(float(calories_match.group().split(":")[1])))
        return servings, calories    
    except Exception as e:
        print(f"Exception {e} occurred while fetching nutrition facts from {nutrition_facts_link}")
        return (0, 0)

def scrape_recipes():
    total_pages = get_number_of_pages()
    recipe_urls = get_recipe_links(total_pages)
    for recipe_url in recipe_urls:
        recipe_data = get_recipes(recipe_url)
        print("Scraped:", recipe_url)
        if recipe_data:
            save_recipe_to_csv(recipe_data)


def save_recipe_to_csv(recipe, filename='../../data/processed/recipes.csv'):
    df = pd.DataFrame([recipe])
    try:
        df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
    except PermissionError:
        print(f"Permission denied for file {filename}")
    except OSError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    scrape_recipes()