import requests
from bs4 import BeautifulSoup
import re

base_url = "https://en.wikipedia.org"

visited_urls = set()

combined_content = ""

def scrape_page(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            content_elements = soup.find_all(["p", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "dl", "table"])

            if content_elements:

                page_content = "\n".join(element.get_text() for element in content_elements)

                global combined_content
                combined_content += page_content + "\n"

                page_title = soup.find("h1").get_text()

                print(f"Content from '{page_title}' added to the combined text.")

            links = soup.find_all("a", href=True)

            wiki_links = [link['href'] for link in links if re.match(r'^/wiki/Finance', link['href'])]

            wiki_links = [f"{base_url}{link}" for link in wiki_links]

            wiki_links = [link for link in wiki_links if link not in visited_urls]

            return wiki_links
        else:
            print(f"Failed to retrieve the Wikipedia page at '{url}'. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred while scraping the page at '{url}': {str(e)}")
        return []

start_url = f"{base_url}/wiki/Finance"

urls_to_scrape = [start_url]

links = [
    "https://en.wikipedia.org/wiki/Finance",
    "https://en.wikipedia.org/wiki/Money",
    "https://en.wikipedia.org/wiki/Tax",
    "https://en.wikipedia.org/wiki/Economic_growth",
    "https://en.wikipedia.org/wiki/Economics",
    "https://en.wikipedia.org/wiki/Investment",
    "https://en.wikipedia.org/wiki/Financial_risk",
    "https://en.wikipedia.org/wiki/Cryptocurrency",
    "https://en.wikipedia.org/wiki/Digital_currency",
]

for current_url in links:
    if current_url not in visited_urls:
        visited_urls.add(current_url)
        new_links = scrape_page(current_url)
        urls_to_scrape.extend(new_links)

with open("wiki.txt", "w", encoding="utf-8") as file:
    file.write(combined_content)

print("Scraping complete. Combined content saved to 'wiki.txt'.")