import os
import re
import json
import time
import requests
import warnings
from pathlib import Path
import openai 

import torch
import numpy as np
import pandas as pd
import spacy
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

API_KEY = "" 
MODEL_NAME = "gpt-oss-120b" 
try:
    cloud_client = openai.OpenAI(api_key=API_KEY, base_url="https://mkp-api.fptcloud.com")
    print("Cloud LLM client configured successfully for judging.")
except Exception as e:
    print(f"Failed to configure OpenAI client for judging: {e}")
    cloud_client = None

print("Loading global models (spaCy)... This may take a moment.")
try:
    SPACY_NLP = spacy.load("en_core_web_lg")
    print("Global models loaded.")
except OSError:
    print("Error: spaCy model 'en_core_web_lg' not found. Please run:")
    print("python -m spacy download en_core_web_lg")
    SPACY_NLP = None

# ==============================================================================
# --- SVO-BERTSum Model Definition and Inference Logic ---
# ==============================================================================

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512

class BERTSummarizer(torch.nn.Module):
    def __init__(self, model_name=PRE_TRAINED_MODEL_NAME):
        super(BERTSummarizer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    def forward(self, input_ids, attention_mask, cls_indices):
        input_ids = input_ids.squeeze(0); attention_mask = attention_mask.squeeze(0)
        outputs = self.bert(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        last_hidden_state = outputs.last_hidden_state.squeeze(0)
        cls_embeddings = last_hidden_state[cls_indices]
        logits = self.classifier(cls_embeddings)
        return torch.sigmoid(logits)

def extract_svo_triples(doc):
    triples = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "VERB":
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                if subjects and objects:
                    for s in subjects:
                        for o in objects:
                            triples.append((s.lemma_.lower(), token.lemma_.lower(), o.lemma_.lower()))
    return triples

def summarize_with_svo_bertsum(text, model, tokenizer, device, probability_threshold=0.5):
    model.eval()
    if not SPACY_NLP: return "spaCy model not loaded."
    try:
        article_sentences = sent_tokenize(text)
    except:
        return ""
    if not article_sentences: return ""

    text_for_bert = ""
    for sent in article_sentences:
        sent_doc = SPACY_NLP(sent)
        triples = extract_svo_triples(sent_doc)
        fact_string = ""
        if triples:
            facts = [" ".join(triple) for triple in triples]
            fact_string = " Facts: " + " ; ".join(facts) + "."
        enriched_input = sent + fact_string
        text_for_bert += enriched_input + " [SEP] [CLS] "
    
    inputs = tokenizer.encode_plus(
        text_for_bert, max_length=MAX_LEN, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    cls_indices = (input_ids.squeeze(0) == tokenizer.cls_token_id).nonzero().flatten()
    if cls_indices.shape[0] == 0: return ""

    with torch.no_grad():
        predictions = model(input_ids, attention_mask, cls_indices).squeeze()
        
    sentence_scores = predictions.cpu().numpy()
    if sentence_scores.ndim == 0:
        sentence_scores = np.array([sentence_scores])

    selected_indices = np.where(sentence_scores > probability_threshold)[0]
    if len(selected_indices) == 0 and len(sentence_scores) > 0:
        selected_indices = [np.argmax(sentence_scores)]
    
    final_indices = sorted(list(selected_indices))
    summary = " ".join([article_sentences[i] for i in final_indices if i < len(article_sentences)])
    return summary

# ==============================================================================
# --- LLM-as-a-Judge and Dataset Functions ---
# ==============================================================================
def llm_as_judge(original_article, candidate_summary):
    if not cloud_client:
        return {"relevance_score": 0, "faithfulness_score": 0, "coherence_score": 0, "conciseness_score": 0, "error": "Cloud client not configured."}

    prompt = (
        "You are an expert evaluator for text summarization systems. Your task is to provide a rigorous, objective "
        "assessment of a generated 'Candidate Summary' based *only* on the provided 'Original Article'.\n\n"
        "Please evaluate the 'Candidate Summary' on four criteria, providing a score from 1 (very poor) "
        "to 5 (excellent) for each. You MUST provide your final answer in JSON format, with integer scores for keys: "
        "relevance_score, faithfulness_score, coherence_score, conciseness_score, and a brief_justification string.\n\n"
        "1.  **Relevance (Coverage):** Does it capture the main ideas? (Score 5 = all key topics present).\n"
        "2.  **Faithfulness (Accuracy):** Is it factually accurate? Any hallucinations? (Score 5 = perfectly faithful).\n"
        "3.  **Coherence (Flow):** Is it well-written and fluent? (Score 5 = reads like a human).\n"
        "4.  **Conciseness (Brevity):** Is it compact and not repetitive? (Score 5 = very concise).\n\n"
        "--- ORIGINAL ARTICLE ---\n{original_article}\n\n"
        "--- CANDIDATE SUMMARY ---\n{candidate_summary}\n\n"
        "--- EVALUATION (JSON) ---"
    ).format(original_article=original_article[:6000], candidate_summary=candidate_summary)
    
    default_response = {"relevance_score": 0, "faithfulness_score": 0, "coherence_score": 0, "conciseness_score": 0}
    try:
        response = cloud_client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.1,
        )
        json_response_text = response.choices[0].message.content
        data = json.loads(json_response_text)
        final_evaluation = {key: int(data.get(key, 0)) for key in default_response}
        return final_evaluation
    except Exception as e:
        print(f"Cloud LLM Judge Error: {e}")
        return default_response

def load_bbc_dataset(base_path):
    print(f"Loading dataset from: {base_path}")
    all_data = []
    articles_path = Path(base_path) / "News Articles"
    summaries_path = Path(base_path) / "Summaries"
    for category_path in articles_path.iterdir():
        if category_path.is_dir():
            category = category_path.name
            for article_file in category_path.glob("*.txt"):
                try:
                    with open(article_file, 'r', encoding='utf-8', errors='ignore') as f: article_content = f.read()
                    summary_file = summaries_path / category / article_file.name
                    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f: summary_content = f.read()
                    all_data.append({"filename": article_file.name, "category": category, "article": article_content, "reference_summary": summary_content})
                except Exception as e:
                    print(f"Skipping file {article_file.name} due to error: {e}")
    return pd.DataFrame(all_data)

# ==============================================================================
# --- Main Evaluation Script ---
# ==============================================================================
def run_svo_bertsum_qualitative_benchmark():
    # --- 1. Load Model and Data ---
    print("Loading fine-tuned SVO-BERTSum model...")
    try:
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        model = BERTSummarizer().to(DEVICE)
        # !! Make sure your model file is named this !!
        model.load_state_dict(torch.load('svo_bertsum_best_model.bin', map_location=DEVICE))
        print("SVO-BERTSum model loaded successfully.")
    except FileNotFoundError:
        print("ERROR: 'svo_bertsum_best_model.bin' not found.")
        return

    DATASET_PATH = "./BBC News Summary"
    if not Path(DATASET_PATH).exists():
        print("ERROR: Dataset not found.")
        return
    df = load_bbc_dataset(DATASET_PATH)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Use a manageable sample for the qualitative benchmark
    sample_df = test_df.sample(n=50, random_state=42) 
    print(f"\nRunning qualitative evaluation on a sample of {len(sample_df)} articles.")

    # --- 2. Evaluation Loop ---
    results = []
    pbar = tqdm(sample_df.itertuples(), total=len(sample_df), desc="Qualitative Eval of SVO-BERTSum")
    for row in pbar:
        article = row.article
        
        # We use the threshold-based summarizer for dynamic length
        candidate = summarize_with_svo_bertsum(article, model, tokenizer, DEVICE, probability_threshold=0.55)
        
        if not candidate: continue
        
        # Get the LLM-as-a-Judge scores (Reference-Free)
        judge_scores = llm_as_judge(article, candidate)
        
        results.append(judge_scores)
        time.sleep(1) # Be kind to the API

    # --- 3. Print Final Report Card ---
    results_df = pd.DataFrame(results)
    
    print("\n\n" + "="*80)
    print("      QUALITATIVE EVALUATION REPORT: Supervised SVO-BERTSum Model      ")
    print("="*80)
    
    print("\n--- Qualitative Metrics (Cloud LLM as Judge, Avg Scores 1-5) ---")
    print(f"Avg Relevance Score:   {results_df['relevance_score'].mean():.2f} / 5.0")
    print(f"Avg Faithfulness Score:  {results_df['faithfulness_score'].mean():.2f} / 5.0")
    print(f"Avg Coherence Score:     {results_df['coherence_score'].mean():.2f} / 5.0")
    print(f"Avg Conciseness Score:   {results_df['conciseness_score'].mean():.2f} / 5.0")

    print("\n" + "="*80)
    print("Evaluation complete.")

if __name__ == "__main__":
    if SPACY_NLP is None:
        print("Exiting because spaCy model could not be loaded.")
    else:
        run_svo_bertsum_qualitative_benchmark()