import requests
from bs4 import BeautifulSoup
import pandas as pd

titles = []
authors = []
dates = []

num_articles = 10

url = 'https://seekingalpha.com/latest-articles'

response = requests.get(url)

if response.status_code == 200:

    soup = BeautifulSoup(response.text, 'html.parser')

    article_elements = soup.find_all('li', class_='list-group-item')

    for article in article_elements[:num_articles]:
        title = article.find('a', class_='a-title').text.strip()
        author = article.find('a', class_='a-author').text.strip()
        date = article.find('time', class_='relative-date')['title'].strip()

        titles.append(title)
        authors.append(author)
        dates.append(date)

    df = pd.DataFrame({'Title': titles, 'Author': authors, 'Date': dates})

    df.to_csv('seeking_alpha_articles.csv', index=False)

    print(f'Scraped {len(df)} articles and saved to seeking_alpha_articles.csv')
else:
    print(f"Failed to retrieve data from the page. Status code: {response.status_code}")