import requests
from bs4 import BeautifulSoup

url = 'https://news.ycombinator.com/'

res = requests.get(url)
soup = BeautifulSoup(res.text, 'html.parser')
titles = soup.select('.titleline > a')  # CSS selector

for title in titles:
    print(title.get_text())