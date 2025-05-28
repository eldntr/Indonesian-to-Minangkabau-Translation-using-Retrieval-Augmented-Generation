def generate_translation_prompt(original_query: str, retrieved_data_list: list) -> str:
    """
    Formats a prompt for an LLM based on the retrieved data.

    Args:
        original_query (str): The original query from the user.
        retrieved_data_list (list): The list of results returned by the retriever.

    Returns:
        str: A prompt string ready to be used by an LLM.
    """
    if not retrieved_data_list:
        return f"No relevant information was found for the query: \"{original_query}\". Please try another query or check the database."

    context_str = "Here is information related to Indonesian words with example sentences in Indonesian and Minangkabau:\n\n"
    for item in retrieved_data_list:
        context_str += f"- For the word \"{item['original_query_word']}\":\n"
        context_str += f"  - Example Sentence (Indonesian): \"{item['retrieved_example']['indonesian']}\"\n"
        context_str += f"  - Example Sentence (Minangkabau): \"{item['retrieved_example']['minangkabau']}\"\n\n"

    prompt = f"""
{context_str}
Your Task:
1. Note that the given words are in Indonesian.
2. Each word has an example sentence in Indonesian and its translation in Minangkabau.
3. Translate the following Indonesian sentence: "{original_query}" into Minangkabau.

Provide only the translated sentence as the output, without any additional text or formatting:
"""
    return prompt