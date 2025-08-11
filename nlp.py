import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score


data = {
    "text": [
        "I love investing in tech startups",
        "The stock market crashed badly",
        "AI is transforming venture capital",
        "This company is going bankrupt",
        "Profits are soaring this quarter",
        "The CEO resigned amid controversy"
    ],
    "label": [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
df.to_csv("sample_data.csv", index=False, encoding='utf-8')

#Tokenization in place
def tokenize(text):
    text = text.lower()
    for char in [".", ",", "!", "?"]:
        text = text.replace(char , "")
    return text.split()
token_list = [tokenize(sentence) for sentence in df["text"]]

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