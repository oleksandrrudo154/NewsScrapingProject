import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Load environment variables from a .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Add it to .env file.")

# Initialize OpenAI language model and embedding model
llm = ChatOpenAI(temperature=0)
embedding_model = OpenAIEmbeddings()

# Set the directory for Chroma vector database
CHROMA_DIR = "chroma_news"
vector_db = Chroma(embedding_function=embedding_model, persist_directory=CHROMA_DIR)

import requests
from bs4 import BeautifulSoup

# Create a requests session for reusing HTTP connections
session = requests.Session()

# Function to fetch article text and title from a URL
def fetch_article(url):
    #Headings to simulate manual search
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    response = session.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: status code {response.status_code}")

    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the article title from <h1> tag
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "No title found"

    # Extract all paragraph texts from <p> tags
    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs)

    return {"title": title, "text": text}

# Function to summarize the article text
def summarize_text(text):
    doc = Document(page_content=text)
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.invoke([doc])
    return summary

# Function to extract main topics from the summary
def extract_topics(summary):
    prompt = f"Identify 3 to 5 key topics for the following news summary:\n\n{summary}"
    response = llm.invoke(prompt)
    return response

# Function to process an article, summarize it, extract topics, and store in vector DB
def process_and_store(url):
    try:
        print(f"Processing: {url}")
        article_data = fetch_article(url)
        summary = summarize_text(article_data["text"])
        topics = extract_topics(summary)

        # Format the content to be stored
        content = f"Title: {article_data['title']}\n\nSummary: {summary}\n\nTopics: {topics}"

        # Add the content and its source URL to the vector database
        vector_db.add_texts(
            texts=[content],
            metadatas=[{"source_url": url}]
        )
        print("Stored successfully.\n")
    except Exception as e:
        print(f"Error processing {url}: {e}")

# Function to perform semantic search on stored content
def semantic_search(query):
    results = vector_db.similarity_search(query, k=3)
    if not results:
        print("No results found.")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content)
        print(f"Source: {doc.metadata.get('source_url')}")

# Main execution block
if __name__ == "__main__":
    # List of article URLs to process
    urls = [
        "https://www.bbc.com/news/articles/ceqg10zqd38o",
        "https://www.bbc.com/news/articles/cx2rd71yy20o",
        "https://www.bbc.com/news/articles/c1j5954edlno",
        "https://www.bbc.com/news/articles/cwy7vl5xn5go",
        "https://www.bbc.com/news/articles/clyz48yypdgo"
    ]

    # Process and store each article
    for url in urls:
        process_and_store(url)

    # Prompt user for a semantic search query
    print("\nSemantic Search")
    user_query = input("Enter your search query: ")
    semantic_search(user_query)
