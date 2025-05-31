def extract_topics(summary, llm):
    """
    Identify main topics from the summary using the LLM.
    """
    prompt = f"Extract 3 to 5 key topics for the following summary:\n{summary}"
    response = llm.invoke(prompt)
    return response