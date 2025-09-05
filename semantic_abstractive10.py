import requests
import json
import sys

def summarize_with_llm_only(text, title, detail_level_str):
    """
    Generates a summary by sending the entire text to a local LLM via LM Studio,
    guided by a detailed prompt that includes the desired level of detail.
    """
    print("\n--- Sending full text to local LLM for pure abstractive summarization ---")
    
    
    # 1. Define the instruction based on the user's chosen detail level.
    if detail_level_str == 'Concise':
        detail_instruction = (
            "The summary should be a very short, high-level overview of 1-2 sentences, "
            "capturing only the absolute main idea or thesis of the article."
        )
    elif detail_level_str == 'Detailed':
        detail_instruction = (
            "The summary should be a comprehensive paragraph of 5-7 sentences. It must cover the main topics, "
            "include the most important supporting details and key examples, and reflect the nuances of the text."
        )
    else: # Balanced
        detail_instruction = (
            "The summary should be a well-rounded paragraph of 3-4 sentences. It must introduce the main topic, "
            "cover the most significant supporting points or examples, and end with a concluding thought."
        )

    # 2. Assemble the final, comprehensive prompt.
    prompt = (
        "You are an expert summarizer and professional editor. Your task is to create a high-quality, "
        "factually accurate, abstractive summary of the following text, which is about '{title}'.\n\n"
        "**Your Instructions:**\n"
        "1.  **Detail Level:** {detail_instruction}\n"
        "2.  **Factual Grounding:** The summary MUST be factually accurate and based ONLY on the information "
        "provided in the text below. Do not add any external information, opinions, or interpretations.\n"
        "3.  **Structure:** The final summary should be a single, cohesive, and fluent paragraph. "
        "It should be well-structured and easy to read.\n\n"
        "--- START OF TEXT TO SUMMARIZE ---\n"
        "{text}\n"
        "--- END OF TEXT TO SUMMARIZE ---\n\n"
        "Now, provide the polished, single-paragraph summary:"
    ).format(title=title, detail_instruction=detail_instruction, text=text)
    
    # 3. Call the LM Studio API (same logic as before)
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "loaded-model",
        "messages": [
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3, # Low temperature for factual tasks
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_json = response.json()
        polished_summary = response_json['choices'][0]['message']['content']
        return polished_summary.strip()
        
    except requests.exceptions.RequestException as e:
        # Error handling remains the same
        error_message = (
            "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "!!! ERROR: Could not connect to LM Studio server.      !!!\n"
            "!!! Please ensure LM Studio is running and the server  !!!\n"
            "!!! has been started on the 'Local Server' tab.        !!!\n"
            f"!!! Details: {e}\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )
        print(error_message)
        return "[ERROR: Could not connect to LLM for summarization.]"

# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    # --- Get User Input ---
    detail_level_options = ['Concise', 'Balanced', 'Detailed']
    DETAIL_LEVEL = ""
    while DETAIL_LEVEL.capitalize() not in detail_level_options:
        DETAIL_LEVEL = input("Enter the detail level for the summary (Options: 'Concise', 'Balanced', 'Detailed'): ")
    DETAIL_LEVEL = DETAIL_LEVEL.capitalize()

    TITLE = input("Enter the title for the text: ")
    print("Enter or paste the text to summarize. Press CTRL+Z and then Enter (Windows) or CTRL+D (Linux/macOS) to finish.")
    TEXT = sys.stdin.read()
    
    # --- Execute the simple, LLM-only pipeline ---
    final_summary = summarize_with_llm_only(TEXT, TITLE, DETAIL_LEVEL)
    
    # --- Display Results ---
    print("\n\n\n=======================================================")
    print(f"        FINAL LLM-Only Summary ('{DETAIL_LEVEL}')        ")
    print("=======================================================")
    print(final_summary)
    print("\n-------------------------------------------------------")
    print(f"STATS: {len(final_summary.split())} words")
    print("-------------------------------------------------------")