from app.scrape import fetch_article
from app.embedding import store_in_vector_db
from app.search import generate_rag_response

urls = [
    "https://www.bbc.com/news/articles/ceqg10zqd38o",
    "https://www.bbc.com/news/articles/cx2rd71yy20o",
    "https://www.bbc.com/news/articles/c1j5954edlno"
]

def process_news(url):
    """
    Processes an article: downloads the text, summarizes it, and stores it in the database.
    """
    article = fetch_article(url)
    if not article:
        print(f"Failed to fetch article: {url}")
        return

    summary = generate_rag_response(article["content"])
    content = f"Title: {article['title']}\n\nSummary: {summary}"
    metadata = {"source_url": url}
    store_in_vector_db(content, metadata)
    print(f"Article processed and stored: {article['title']}")

if __name__ == "__main__":
    # 1. Process articles
    for url in urls:
        process_news(url)

    # 2. User query input
    user_query = input("Enter your query: ")
    rag_response = generate_rag_response(user_query)

    # 3. Output the response
    print("\nRAG Response:")
    print(rag_response)
