import os

from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from app.embedding import search_vector_db

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY не найден в .env или окружении")

# Инициализация модели для генерации
llm = OpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)

# Шаблон для генерации ответов (RAG)
RAG_PROMPT_TEMPLATE = """
You are an AI assistant tasked with helping the user. Use the following context to answer their question.

### Context:
{context}

### User Query:
{query}

### Answer:
"""

def generate_rag_response(query):
    """
    RAG (Retrieval-Augmented Generation) для генерации ответа на основе данных из базы.
    """
    # Извлекаем релевантные документы
    relevant_docs = search_vector_db(query, top_k=3)

    if not relevant_docs:
        return "Извините, я не нашёл релевантную информацию для вашего запроса."

    # Генерируем контекст для модели
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Формируем шаблон модели
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=RAG_PROMPT_TEMPLATE
    ).format(query=query, context=context)

    try:
        response = llm.predict(prompt)  # Генерация ответа на основе контекста
        return response
    except Exception as e:
        return f"Ошибка при генерации ответа: {e}"