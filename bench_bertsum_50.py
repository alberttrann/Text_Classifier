import os
import re
import json
import time
import requests
import warnings
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
from tqdm import tqdm

API_KEY = "" 
BASE_URL = "https://mkp-api.fptcloud.com"
MODEL_NAME = "gpt-oss-120b"

# Create the OpenAI client for the cloud model
try:
    import openai
    cloud_client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("Cloud LLM client configured successfully.")
except Exception as e:
    print(f"Failed to configure OpenAI client: {e}")

# --- Global Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# --- BERTSum Model Definition and Inference Logic ---
# ==============================================================================

PRE_TRAINED_MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
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
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        outputs = self.bert(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        last_hidden_state = outputs.last_hidden_state.squeeze(0)
        cls_embeddings = last_hidden_state[cls_indices]
        logits = self.classifier(cls_embeddings)
        return torch.sigmoid(logits)

def summarize_with_bertsum(text, model, tokenizer, device, probability_threshold=0.5):
    model.eval()
    try:
        article_sentences = sent_tokenize(text)
    except:
        return "Could not process text."
    if not article_sentences: return ""

    text_for_bert = " [SEP] [CLS] ".join(article_sentences)
    
    inputs = tokenizer.encode_plus(
        text_for_bert, max_length=MAX_LEN, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    cls_indices = (input_ids.squeeze(0) == tokenizer.cls_token_id).nonzero().flatten()

    with torch.no_grad():
        predictions = model(input_ids, attention_mask, cls_indices).squeeze()
        
    sentence_scores = predictions.cpu().numpy()
    if sentence_scores.ndim == 0:
        sentence_scores = np.array([sentence_scores])

    ## NEW LOGIC ##
    # Select all sentences whose predicted probability is above the threshold.
    selected_indices = np.where(sentence_scores > probability_threshold)[0]
    
    # Fallback: if no sentence meets the threshold, pick the single best one.
    if len(selected_indices) == 0:
        selected_indices = [np.argmax(sentence_scores)]
    
    # We no longer need to sort by argsort, but we do need to sort the final indices
    # to maintain original order.
    final_indices = sorted(list(selected_indices))
    
    summary = " ".join([article_sentences[i] for i in final_indices if i < len(article_sentences)])
    return summary

# ==============================================================================
# --- Evaluation Metrics ---
# ==============================================================================

class NLI_Scorer:
    def __init__(self, nli_model_name='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'):
        print(f"Loading NLI model: {nli_model_name}...")
        nli_device = 0 if torch.cuda.is_available() else -1
        self.nli_pipeline = hf_pipeline("text-classification", model=nli_model_name, device=nli_device)
        self.semantic_search_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        print("NLI Scorer initialized.")

    def score(self, article_text, summary_text):
        if not article_text or not summary_text: return 0.0, 0.0
        try:
            article_sentences = sent_tokenize(article_text)
            summary_sentences = sent_tokenize(summary_text)
            if not article_sentences or not summary_sentences: return 0.0, 0.0
        except Exception: return 0.0, 0.0

        article_embeddings = self.semantic_search_model.encode(article_sentences, convert_to_tensor=True)
        entailment_scores, contradiction_scores = [], []

        for summary_sent in summary_sentences:
            summary_sent_embedding = self.semantic_search_model.encode(summary_sent, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(summary_sent_embedding, article_embeddings)[0]
            premise_idx = torch.argmax(similarities).item()
            premise_sent = article_sentences[premise_idx]
            
            nli_results = self.nli_pipeline(f"{premise_sent}</s></s>{summary_sent}")
            
            entailment_score, contradiction_score = 0.0, 0.0
            for result in nli_results:
                if result['label'] == 'entailment': entailment_score = result['score']
                elif result['label'] == 'contradiction': contradiction_score = result['score']
            entailment_scores.append(entailment_score)
            contradiction_scores.append(contradiction_score)

        return np.mean(entailment_scores), np.max(contradiction_scores)

def llm_as_judge(original_article, candidate_summary):
    """
    Uses the configured cloud LLM (e.g., gpt-oss-20b) to score a summary.
    """
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
            # Note: response_format is specific to OpenAI's newer models.
            # If your FPT endpoint doesn't support it, you may need to remove this line.
            # It's a hint to the model, but not always required for it to produce JSON.
            # response_format={"type": "json_object"} 
        )
        json_response_text = response.choices[0].message.content
        data = json.loads(json_response_text)
        final_evaluation = {key: int(data.get(key, 0)) for key in default_response}
        return final_evaluation
    except Exception as e:
        print(f"Cloud LLM Judge Error: {e}")
        return default_response

# ==============================================================================
# --- Main Evaluation Script ---
# ==============================================================================

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

def run_bertsum_evaluation():
    # --- 1. Load Model and Data ---
    print("Loading fine-tuned BERTSum model...")
    try:
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        model = BERTSummarizer().to(DEVICE)
        model.load_state_dict(torch.load('bertsum_best_model.bin', map_location=DEVICE))
        print("BERTSum model loaded successfully.")
    except FileNotFoundError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: 'bertsum_best_model.bin' not found.               !!!")
        print("!!! Please place your fine-tuned model file in the same    !!!")
        print("!!! directory as this script.                              !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    DATASET_PATH = "./BBC News Summary"
    if not Path(DATASET_PATH).exists():
        print("ERROR: Dataset not found.")
        return
    df = load_bbc_dataset(DATASET_PATH)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    sample_df = test_df.sample(n=50, random_state=42) # Use a 20-article sample
    print(f"\nRunning full evaluation on a sample of {len(sample_df)} articles.")

    # --- 2. Initialize Scorers ---
    rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    nli_eval = NLI_Scorer()

    # --- 3. Evaluation Loop ---
    results = []
    pbar = tqdm(sample_df.itertuples(), total=len(sample_df), desc="Evaluating BERTSum")
    for row in pbar:
        article = row.article
        reference = row.reference_summary
        
        # Generate summary with dynamic length to match reference for fairness
        num_sents = len(sent_tokenize(reference))
        ## MODIFIED ## - Call the new, threshold-based summarizer
        candidate = summarize_with_bertsum(article, model, tokenizer, DEVICE, probability_threshold=0.5)
        
        if not candidate: continue
        
        # ROUGE
        rouge_scores = rouge_eval.score(reference, candidate)
        
        # BERTScore
        _, _, bert_f1 = bert_scorer([candidate], [reference], lang="en", verbose=False)
        
        # NLI Score
        nli_entail, nli_contra = nli_eval.score(article, candidate)

        # LLM-as-a-Judge Score
        judge_scores = llm_as_judge(article, candidate)
        
        results.append({
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'bertscore_f1': bert_f1.mean().item(),
            'nli_entailment': nli_entail,
            'nli_contradiction': nli_contra,
            'judge_relevance': judge_scores['relevance_score'],
            'judge_faithfulness': judge_scores['faithfulness_score'],
            'judge_coherence': judge_scores['coherence_score'],
            'judge_conciseness': judge_scores['conciseness_score'],
        })
        time.sleep(1) # Small delay for API calls

    # --- 4. Print Final Report Card ---
    results_df = pd.DataFrame(results)
    
    print("\n\n" + "="*80)
    print("                    HOLISTIC EVALUATION REPORT: Fine-tuned BERTSum                    ")
    print("="*80)
    
    print("\n--- Quantitative Metrics (Average Scores) ---")
    print(f"Avg ROUGE-1 F1:      {results_df['rouge1'].mean():.4f}")
    print(f"Avg ROUGE-2 F1:      {results_df['rouge2'].mean():.4f}")
    print(f"Avg ROUGE-L F1:      {results_df['rougeL'].mean():.4f}")
    print("-" * 40)
    print(f"Avg BERTScore F1:    {results_df['bertscore_f1'].mean():.4f}")
    print("-" * 40)
    print(f"Avg NLI Entailment:  {results_df['nli_entailment'].mean():.4f} (Higher is better)")
    print(f"Avg NLI Contradiction: {results_df['nli_contradiction'].mean():.4f} (Lower is better)")
    
    print("\n\n--- Qualitative Metrics (LLM-as-a-Judge, Avg Scores 1-5) ---")
    print(f"Avg Relevance Score:   {results_df['judge_relevance'].mean():.2f} / 5.0")
    print(f"Avg Faithfulness Score:  {results_df['judge_faithfulness'].mean():.2f} / 5.0")
    print(f"Avg Coherence Score:     {results_df['judge_coherence'].mean():.2f} / 5.0")
    print(f"Avg Conciseness Score:   {results_df['judge_conciseness'].mean():.2f} / 5.0")

    print("\n" + "="*80)
    print("Evaluation complete.")


if __name__ == "__main__":
    # Ensure your fine-tuned model 'bertsum_best_model.bin' is in the same directory.
    # Ensure your LM Studio server is running for the LLM-as-a-Judge evaluation.
    run_bertsum_evaluation()