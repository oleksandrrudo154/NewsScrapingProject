import requests
from bs4 import BeautifulSoup

# Function to fetch article text and title from a URL
def fetch_article(url):
    """
    Fetch the article title and content from the provided URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch the page. Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No title found"
    paragraphs = soup.find_all("p")
    content = "\n".join(p.get_text(strip=True) for p in paragraphs)

    return {"title": title, "content": content}