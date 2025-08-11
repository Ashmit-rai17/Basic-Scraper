import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score
import requests
from dotenv import load_dotenv
import os
import random
load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")
url = f"https://finnhub.io/api/v1/news?category=general&token={API_KEY}"
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Error fetching data: {response.status_code}")
data = response.json()
articles = []
for article in data:
    articles.append({"summary": article.get("summary", "") , "label" : random.randint(0,1)} ) 

df = pd.DataFrame(articles)
df.to_csv("sample_data.csv", index=False, encoding='utf-8')

#Tokenization in place
def tokenize(text):
    text = text.lower()
    for char in [".", ",", "!", "?"]:
        text = text.replace(char , "")
    return text.split()
token_list = [tokenize(sentence) for sentence in df["summary"]]

vocab = {}
for tokens in token_list:
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab) #assigns each unique id to a word
print("Vocubalary: " , vocab)

#Now , implementing bag of words
def text_to_vector(tokens , vocab):
    vector = np.zeros(len(vocab))
    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1
    return vector

X = np.array([text_to_vector(tokens , vocab) for tokens in token_list])
y = df["label"].values.ravel()

print("Feature Matrix : \n" , X)

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train , y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
print("The accuracy score is: " , accuracy*100 , "%")
