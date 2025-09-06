import os
import re
import warnings
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import spacy
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
from tqdm import tqdm

# --- Global Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Global Models (loaded once) ---
print("Loading global models (spaCy, SBERT for NLI)...")
SPACY_NLP = spacy.load("en_core_web_lg")
SBERT_MODEL = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=DEVICE)
NLI_PIPELINE = hf_pipeline("text-classification", model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', device=0 if torch.cuda.is_available() else -1)
print("Global models loaded.")


# ==============================================================================
# --- SVO-BERTSum Model Definition and Inference Logic ---
# ==============================================================================

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512

class BERTSummarizer(torch.nn.Module):
    """
    The BERTSum model architecture. This must match the architecture
    that was used during training.
    """
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

def summarize_with_svo_bertsum(text, model, tokenizer, device, probability_threshold=0.5):
    """
    Summarizes text using the SVO-enriched input format that the model was
    trained on.
    """
    model.eval()
    try:
        article_sentences = sent_tokenize(text)
    except:
        return ""
    if not article_sentences: return ""

    # --- SVO ENRICHMENT DURING INFERENCE ---
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
    # ----------------------------------------
    
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
                    all_data.append({"article": article_content, "reference_summary": summary_content})
                except Exception as e:
                    print(f"Skipping file {article_file.name} due to error: {e}")
    return pd.DataFrame(all_data)

def run_svo_bertsum_benchmark():
    # --- 1. Load Model and Data ---
    print("Loading fine-tuned SVO-BERTSum model...")
    try:
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        model = BERTSummarizer().to(DEVICE)
        # !! Make sure your model file is named this !!
        model.load_state_dict(torch.load('svo_bertsum_best_model.bin', map_location=DEVICE))
        print("SVO-BERTSum model loaded successfully.")
    except FileNotFoundError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: 'svo_bertsum_best_model.bin' not found.             !!!")
        print("!!! Please place your SVO-trained model file in the same     !!!")
        print("!!! directory as this script.                                !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    DATASET_PATH = "./BBC News Summary"
    if not Path(DATASET_PATH).exists():
        print("ERROR: Dataset not found.")
        return
    df = load_bbc_dataset(DATASET_PATH)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nRunning SVO-BERTSum quantitative benchmark on {len(test_df)} articles.")

    # --- 2. Generation Loop ---
    candidates = []
    references = list(test_df['reference_summary'])
    articles = list(test_df['article'])

    pbar = tqdm(range(len(test_df)), desc="Generating Summaries with SVO-BERTSum")
    for i in pbar:
        # For this benchmark, we'll use a fixed probability threshold
        candidate = summarize_with_svo_bertsum(articles[i], model, tokenizer, DEVICE, probability_threshold=0.5)
        candidates.append(candidate)

    # --- 3. Batch Score Calculation ---
    rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_results, nli_results = [], []

    print("\nCalculating ROUGE and NLI scores...")
    for i in tqdm(range(len(candidates)), desc="Scoring (ROUGE & NLI)"):
        if candidates[i] and references[i]:
            rouge_scores = rouge_eval.score(references[i], candidates[i])
            rouge_results.append(rouge_scores)
            
            nli_entail, nli_contra = nli_score(articles[i], candidates[i])
            nli_results.append({'entailment': nli_entail, 'contradiction': nli_contra})

    print("Calculating BERTScore (this may take a while)...")
    # Filter out empty strings before passing to bert_scorer
    filtered_candidates = [c for c in candidates if c]
    filtered_references = [r for c, r in zip(candidates, references) if c]
    if filtered_candidates:
        _, _, bert_f1_scores = bert_scorer(filtered_candidates, filtered_references, lang="en", verbose=True, batch_size=16)
        bertscore_f1 = bert_f1_scores.mean().item()
    else:
        bertscore_f1 = 0.0
    
    # --- 4. Print Final Report Card ---
    rouge1 = np.mean([r['rouge1'].fmeasure for r in rouge_results]) if rouge_results else 0.0
    rouge2 = np.mean([r['rouge2'].fmeasure for r in rouge_results]) if rouge_results else 0.0
    rougeL = np.mean([r['rougeL'].fmeasure for r in rouge_results]) if rouge_results else 0.0
    nli_entailment = np.mean([n['entailment'] for n in nli_results]) if nli_results else 0.0
    nli_contradiction = np.mean([n['contradiction'] for n in nli_results]) if nli_results else 0.0
    
    print("\n\n" + "="*80)
    print("         HOLISTIC EVALUATION REPORT: Supervised SVO-BERTSum Model         ")
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
    run_svo_bertsum_benchmark()