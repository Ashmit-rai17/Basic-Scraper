import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

API_KEY = os.getenv("FINNHUB_API_KEY")

url = "https://finnhub.io/api/v1/news" #Url for fetching news articles
params = {
    "category": "general",   # parameter to specify the category of news
    "token": API_KEY
}
dict_articles = {}
response = requests.get(url, params=params)

if response.status_code == 200: # response code 200 means that it worked
    data = response.json() # store it all into a json file
    for article in data:
        dict_articles[article["id"]] = {
            "title": article["headline"],
            "source": article["source"],
            "url": article["url"],
            "published_at": article["datetime"]
        }
        print(f"Title: {article['headline']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}")
        print(f"Published At: {article['datetime']}")
        print("-" * 80)
else:
    print("Error:", response.status_code, response.text)
csv_file = pd.DataFrame.from_dict(dict_articles, orient="index")
csv_file.to_csv("news_articles.csv", index=False)