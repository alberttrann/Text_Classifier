"""
summarizers/llm_only.py

This script contains the implementation for the pure abstractive summarization
model, which sends the entire text to a local Large Language Model.

Models:
1. LLM_Only (Balanced and Detailed modes)
"""

import requests
import json

def llm_only_summarize(text, title, detail_level='Balanced'):
    """
    Generates a summary by sending the entire text to a local LLM via LM Studio,
    guided by a detailed prompt that includes the desired level of detail.

    Returns:
        - summary (str): The final summary text from the LLM.
        - details (dict): A dictionary containing the prompt sent to the LLM for transparency.
    """
    print("\n--- Sending full text to local LLM for pure abstractive summarization ---")

    # --- 1. The "Golden Prompt" ---
    
    # A. Define the instruction based on the user's chosen detail level.
    if detail_level == 'Detailed':
        detail_instruction = (
            "The summary should be a comprehensive paragraph of 5-7 sentences. It must cover the main topics, "
            "include the most important supporting details and key examples, and reflect the nuances of the text."
        )
    else: # Balanced is the default
        detail_instruction = (
            "The summary should be a well-rounded paragraph of 3-4 sentences. It must introduce the main topic, "
            "cover the most significant supporting points or examples, and end with a concluding thought."
        )

    # B. Assemble the final, comprehensive prompt.
    prompt = (
        "You are an expert summarizer and professional editor. Your task is to create a high-quality, "
        "factually accurate, abstractive summary of the following text, which is about '{title}'.\n\n"
        "**Your Instructions:**\n"
        "1.  **Detail Level:** {detail_instruction}\n"
        "2.  **Factual Grounding:** The summary MUST be factually accurate and based ONLY on the information "
        "provided in the text below. Do not add any external information, opinions, or interpretations.\n"
        "3.  **Structure:** The final summary should be a single, cohesive, and fluent paragraph.\n\n"
        "--- START OF TEXT TO SUMMARIZE ---\n"
        "{text}\n"
        "--- END OF TEXT TO SUMMARIZE ---\n\n"
        "Now, provide the polished, single-paragraph summary:"
    ).format(title=title, detail_instruction=detail_instruction, text=text)
    
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "loaded-model", # This is a placeholder, LM Studio uses the model you've loaded
        "messages": [
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3, # Low temperature for factual tasks
        "max_tokens": 512,  
    }
    
    summary = ""
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120) # Increased timeout
        response.raise_for_status()
        response_json = response.json()
        summary = response_json['choices'][0]['message']['content'].strip()
        
    except requests.exceptions.RequestException as e:
        error_message = (
            "!!! ERROR: Could not connect to LM Studio server. "
            "Please ensure LM Studio is running and the server has been started. "
            f"Details: {e}"
        )
        print(error_message)
        summary = f"[ERROR: {error_message}]"
    
    # 3. Package details for visualization
    details = {
        "prompt_sent_to_llm": prompt
    }
    
    return summary, details