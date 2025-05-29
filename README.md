News Scraping & Semantic Search with GenAI
A Python project for automated news article processing, including scraping, summarization, topic extraction, and semantic search powered by OpenAI GPT models and vector databases.

🚀 Features
🌐 Web Scraping: Automatically fetches news articles using requests and BeautifulSoup.

✨ Summarization: Generates concise summaries using GPT-based language models.

🧠 Topic Extraction: Identifies 3–5 key topics from each article's summary.

📦 Vector Storage: Stores processed content in a vector database (Chroma) for efficient querying.

🔍 Semantic Search: Allows context-aware searching of news articles.

📦 Installation
1. Clone the Repository
git clone https://github.com/username/NewsScrapingProject.git
cd NewsScrapingProject
2. Install Dependencies
Make sure you have Python 3.8+ installed.


pip install -r requirements.txt
3. Set Up API Key
Create a .env file in the root directory with your OpenAI API key:


OPENAI_API_KEY=your_openai_key_here
🛠 Usage
Run the main script to scrape, process, and store news articles:


python src/main.py
When prompted, enter your search query:


Enter a search query: Artificial Intelligence in Medicine
You'll receive a list of the most relevant articles, including their summaries and original source links.

🧰 Tech Stack
Python 3.8+

LangChain – for chaining LLM tasks

OpenAI GPT – for summarization & topic extraction

BeautifulSoup – for HTML parsing

Chroma – vector database for semantic search

📄 Project Structure
.env # API key config

requirements.txt # Python dependencies

main.py # Main processing and search logic

chroma_news/ # Vector database storage
