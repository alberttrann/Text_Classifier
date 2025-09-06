import os
import re
import warnings
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import spacy
import xgboost as xgb
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print("Loading global models (spaCy, SBERT for NLI)...")
SPACY_NLP = spacy.load("en_core_web_lg")
SBERT_MODEL = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=DEVICE)
NLI_PIPELINE = hf_pipeline("text-classification", model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', device=0 if torch.cuda.is_available() else -1)
print("Global models loaded.")

# ==============================================================================
# --- XGBoost Model Inference Logic ---
# ==============================================================================

def extract_svo_triples(doc):
    """Extracts Subject-Verb-Object triples from a spaCy Doc object."""
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

def summarize_with_svo_xgb(text, model, num_sents=3):
    """
    Summarizes a new article using the trained SVO-XGBoost model.
    """
    try:
        article_sentences = sent_tokenize(text)
    except:
        return ""
    if not article_sentences: return ""

    article_doc = SPACY_NLP(text)
    article_kg = Counter(extract_svo_triples(article_doc))
    
    sentence_features = []
    for i, sentence_text in enumerate(article_sentences):
        sentence_doc = SPACY_NLP(sentence_text)
        sentence_triples = extract_svo_triples(sentence_doc)
        freqs = [article_kg.get(triple, 0) for triple in sentence_triples]
        
        features = {
            'sentence_position': i / len(article_sentences),
            'sentence_length': len([token for token in sentence_doc if not token.is_punct]),
            'numerical_data_count': len(re.findall(r'\d+', sentence_text)),
            'proper_noun_count': len([token for token in sentence_doc if token.pos_ == "PROPN"]),
            'num_triples_in_sentence': len(sentence_triples),
            'avg_triple_frequency': np.mean(freqs) if freqs else 0,
            'max_triple_frequency': np.max(freqs) if freqs else 0,
        }
        sentence_features.append(features)
        
    features_df = pd.DataFrame(sentence_features)
    
    # Ensure column order matches the training order
    feature_order = [
        'sentence_position', 'sentence_length', 'numerical_data_count', 
        'proper_noun_count', 'num_triples_in_sentence', 
        'avg_triple_frequency', 'max_triple_frequency'
    ]
    features_df = features_df[feature_order]

    predictions = model.predict_proba(features_df)[:, 1]
    
    num_to_select = min(num_sents, len(article_sentences))
    top_indices = np.argsort(predictions)[-num_to_select:]
    top_indices.sort()
    
    summary = " ".join([article_sentences[i] for i in top_indices])
    return summary

# ==============================================================================
# --- Evaluation Metrics ---
# ==============================================================================

def nli_score(article_text, summary_text):
    if not article_text or not summary_text: return 0.0, 0.0
    try:
        article_sentences = sent_tokenize(article_text)
        summary_sentences = sent_tokenize(summary_text)
        if not article_sentences or not summary_sentences: return 0.0, 0.0
    except Exception: return 0.0, 0.0
    article_embeddings = SBERT_MODEL.encode(article_sentences, convert_to_tensor=True, show_progress_bar=False)
    entailment_scores, contradiction_scores = [], []
    for summary_sent in summary_sentences:
        summary_sent_embedding = SBERT_MODEL.encode(summary_sent, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.pytorch_cos_sim(summary_sent_embedding, article_embeddings)[0]
        premise_idx = torch.argmax(similarities).item()
        premise_sent = article_sentences[premise_idx]
        nli_results = NLI_PIPELINE(f"{premise_sent}</s></s>{summary_sent}")
        entailment_score, contradiction_score = 0.0, 0.0
        for result in nli_results:
            if result['label'] == 'entailment': entailment_score = result['score']
            elif result['label'] == 'contradiction': contradiction_score = result['score']
        entailment_scores.append(entailment_score)
        contradiction_scores.append(contradiction_score)
    return np.mean(entailment_scores) if entailment_scores else 0.0, np.max(contradiction_scores) if contradiction_scores else 0.0

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

def run_svo_xgb_benchmark():
    # --- 1. Load Model and Data ---
    print("Loading fine-tuned SVO-XGBoost model...")
    try:
        model = xgb.XGBClassifier()
        model.load_model("svo_xgb_summarizer.json")
        print("SVO-XGBoost model loaded successfully.")
    except (xgb.core.XGBoostError, FileNotFoundError):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: 'svo_xgb_summarizer.json' not found or is invalid. !!!")
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
    print(f"\nRunning SVO-XGBoost benchmark on {len(test_df)} articles.")

    # --- 2. Generation Loop ---
    candidates = []
    references = list(test_df['reference_summary'])
    articles = list(test_df['article'])

    pbar = tqdm(range(len(test_df)), desc="Generating Summaries with SVO-XGBoost")
    for i in pbar:
        # Fair Length Control: determine target length from reference
        try:
            target_len = len(sent_tokenize(references[i]))
        except:
            target_len = 3 
            
        candidate = summarize_with_svo_xgb(articles[i], model, num_sents=target_len)
        candidates.append(candidate)

    # --- 3. Batch Score Calculation ---
    rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_results, nli_results = [], []

    print("\nCalculating ROUGE and NLI scores...")
    for i in tqdm(range(len(candidates)), desc="Scoring"):
        if candidates[i] and references[i]:
            rouge_scores = rouge_eval.score(references[i], candidates[i])
            rouge_results.append(rouge_scores)
            
            nli_entail, nli_contra = nli_score(articles[i], candidates[i])
            nli_results.append({'entailment': nli_entail, 'contradiction': nli_contra})

    print("Calculating BERTScore (this may take a while)...")
    _, _, bert_f1_scores = bert_scorer(candidates, references, lang="en", verbose=True, batch_size=16)
    
    # --- 4. Print Final Report Card ---
    rouge1 = np.mean([r['rouge1'].fmeasure for r in rouge_results])
    rouge2 = np.mean([r['rouge2'].fmeasure for r in rouge_results])
    rougeL = np.mean([r['rougeL'].fmeasure for r in rouge_results])
    bertscore_f1 = bert_f1_scores.mean().item()
    nli_entailment = np.mean([n['entailment'] for n in nli_results])
    nli_contradiction = np.mean([n['contradiction'] for n in nli_results])

    print("\n\n" + "="*80)
    print("         HOLISTIC EVALUATION REPORT: Supervised SVO-XGBoost Model         ")
    print("="*80)
    print("\n--- Quantitative Metrics (Average Scores on Full Test Set) ---")
    print(f"Avg ROUGE-1 F1:      {rouge1:.4f}")
    print(f"Avg ROUGE-2 F1:      {rouge2:.4f}")
    print(f"Avg ROUGE-L F1:      {rougeL:.4f}")
    print("-" * 40)
    print(f"Avg BERTScore F1:    {bertscore_f1:.4f}")
    print("-" * 40)
    print(f"Avg NLI Entailment:  {nli_entailment:.4f} (Higher is better)")
    print(f"Avg NLI Contradiction: {nli_contradiction:.4f} (Lower is better)")
    print("\n" + "="*80)
    print("Evaluation complete.")

if __name__ == "__main__":
    run_svo_xgb_benchmark()