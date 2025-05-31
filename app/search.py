from langchain.prompts import PromptTemplate
from app.embedding import search_vector_db
from langchain_openai import ChatOpenAI

# Initialize the generative model
llm = ChatOpenAI(temperature=0)

# Template for RAG queries
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the context below to generate a response to the user's query.

### Context:
{context}

### User Query:
{query}

### Answer:
"""


def generate_rag_response(query):
    """
    Performs a RAG approach:
    1. Extracts relevant documents from the database.
    2. Uses them as context to generate a response using LLM.
    """
    # Search for relevant documents
    relevant_docs = search_vector_db(query, top_k=3)

    # If no documents are found, return a message about empty results
    if not relevant_docs:
        return "Sorry, I couldn't find relevant information for your query."

    # Form context from the found documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Generate final prompt for the LLM
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=RAG_PROMPT_TEMPLATE
    ).format(query=query, context=context)

    try:
        response = llm.predict(prompt)  # Generate response based on context
        return response
    except Exception as e:
        return f"Error during RAG response generation: {e}"
