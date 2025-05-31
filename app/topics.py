def extract_topics(summary, llm):
    try:
        prompt = f"Extract 3 to 5 key topics for the following summary:\n{summary}"
        response = llm.invoke(prompt)  # Generate response based on context
        return response
    except Exception as e:
        if "token limit" in str(e).lower():
            return "Error: Input exceeds the token limit of the model."
        return f"Error during response generation: {e}"

