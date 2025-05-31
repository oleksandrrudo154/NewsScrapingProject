from langchain.prompts import PromptTemplate
from app.embedding import search_vector_db
from langchain_openai import ChatOpenAI
import tiktoken

# Token limit for the model
MAX_TOKENS = 4096

# Initialize the generative model
llm = ChatOpenAI(temperature=0)

# Prompt template for RAG
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the context below to generate a response to the user's query.

### Context:
{context}

### User Query:
{query}

### Answer:
"""

def truncate_context(context, query):
    """
    Truncates the context so that the combined length of the context and query
    does not exceed the model's token limit.
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    query_tokens = len(encoding.encode(query))

    context_tokens = encoding.encode(context)
    max_context_tokens = MAX_TOKENS - query_tokens - 500  # Reserve tokens for the response

    if len(context_tokens) > max_context_tokens:
        truncated_context = encoding.decode(context_tokens[:max_context_tokens])
        return truncated_context
    return context

def generate_rag_response(query):
    """
    Implements the Retrieval-Augmented Generation (RAG) approach:
    1. Retrieves relevant documents from the vector database.
    2. Uses them as context to generate a response with the LLM.
    """
    # Retrieve relevant documents
    relevant_docs = search_vector_db(query, top_k=3)

    # Return a message if no relevant documents are found
    if not relevant_docs:
        return "Sorry, I couldn't find relevant information for your query."

    # Construct and truncate the context if necessary
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    context = truncate_context(context, query)

    # Generate the final prompt for the LLM
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=RAG_PROMPT_TEMPLATE
    ).format(query=query, context=context)

    try:
        response = llm.invoke(prompt)  # Generate the response based on the context
        return response
    except Exception as e:
        if "token limit" in str(e).lower():
            return "Error: Input exceeds the token limit of the model."
        return f"Error during RAG response generation: {e}"
