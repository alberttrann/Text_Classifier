import os
import re
import json
import time
import requests
import warnings
from pathlib import Path
import numpy as np
import openai

import pandas as pd
import spacy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

CLOUD_API_KEY = "" 
CLOUD_MODEL_NAME = "gpt-oss-120b"  
CLOUD_BASE_URL = "https://mkp-api.fptcloud.com"  

try:
    cloud_client = openai.OpenAI(
        api_key=CLOUD_API_KEY,
        base_url=CLOUD_BASE_URL
    )
    print("Cloud LLM client for judging configured successfully.")
except Exception as e:
    print(f"Failed to configure OpenAI client for judging: {e}")
    cloud_client = None

print("Loading global models (spaCy)... This may take a moment.")
try:
    SPACY_NLP = spacy.load("en_core_web_lg")
    print("Global models loaded.")
except OSError:
    print("Error: spaCy model 'en_core_web_lg' not found. Please install it with:")
    print("python -m spacy download en_core_web_lg")
    SPACY_NLP = None

# ==============================================================================
# --- SVO-XGBoost Model and Helper Definitions ---
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
    if not SPACY_NLP:
        print("Error: spaCy model not loaded. Cannot generate summary.")
        return ""
        
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
    
    feature_order = [
        'sentence_position', 'sentence_length', 'numerical_data_count', 
        'proper_noun_count', 'num_triples_in_sentence', 
        'avg_triple_frequency', 'max_triple_frequency'
    ]
    features_df = features_df[feature_order]

    try:
        predictions = model.predict_proba(features_df)[:, 1]
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return ""
    
    num_to_select = min(num_sents, len(article_sentences))
    top_indices = np.argsort(predictions)[-num_to_select:]
    top_indices.sort()
    
    summary = " ".join([article_sentences[i] for i in top_indices])
    return summary

# ==============================================================================
# --- LLM-as-a-Judge and Dataset Functions ---
# ==============================================================================
def llm_as_judge(original_article, candidate_summary):
    """
    Uses a cloud LLM to score a candidate summary based ONLY on the original article.
    """
    if not cloud_client:
        return {"relevance_score": 0, "faithfulness_score": 0, "coherence_score": 0, "conciseness_score": 0, "error": "Cloud client not configured."}

    prompt = (
        "You are an expert evaluator for text summarization systems. Your task is to provide a rigorous, objective "
        "assessment of a generated 'Candidate Summary' based *only* on the provided 'Original Article'. "
        "Do not use any external knowledge.\n\n"
        "Please evaluate the 'Candidate Summary' on four criteria, providing an integer score from 1 (very poor) "
        "to 5 (excellent) for each. You MUST provide your final answer in JSON format, with keys: relevance_score, "
        "faithfulness_score, coherence_score, conciseness_score, and a brief_justification string.\n\n"
        "--- ORIGINAL ARTICLE ---\n{original_article}\n\n"
        "--- CANDIDATE SUMMARY ---\n{candidate_summary}\n\n"
        "--- EVALUATION (Return ONLY the raw JSON object) ---"
    ).format(original_article=original_article[:6000], candidate_summary=candidate_summary)
    
    default_response = {"relevance_score": 0, "faithfulness_score": 0, "coherence_score": 0, "conciseness_score": 0, "brief_justification": "LLM call failed.", "error": "Unknown"}
    
    try:
        response = cloud_client.chat.completions.create(
            model=CLOUD_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        
        json_response_text = response.choices[0].message.content.strip()
        print(f"Raw LLM response: {json_response_text[:200]}...")  
        
        try:
            data = json.loads(json_response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', json_response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise json.JSONDecodeError("No valid JSON found", json_response_text, 0)
        
        final_evaluation = {}
        score_keys = ['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']
        for key in score_keys:
            try:
                score = data.get(key, 0)
                # Handle various formats
                if isinstance(score, str):
                    score = int(re.search(r'\d+', score).group()) if re.search(r'\d+', score) else 0
                final_evaluation[key] = int(score)
            except (ValueError, TypeError, AttributeError):
                final_evaluation[key] = 0
        final_evaluation["brief_justification"] = str(data.get("brief_justification", "N/A"))
        
        print(f"Parsed scores: {final_evaluation}") 

    except Exception as e:
        print(f"LLM judge error: {e}")  
        default_response["error"] = str(e)
        return default_response

def load_bbc_dataset(base_path):
    print(f"Loading dataset from: {base_path}")
    all_data = []
    articles_path = Path(base_path) / "News Articles"
    summaries_path = Path(base_path) / "Summaries"
    
    if not articles_path.exists() or not summaries_path.exists():
        print(f"Error: Dataset paths do not exist. Expected structure:")
        print(f"  {base_path}/News Articles/")
        print(f"  {base_path}/Summaries/")
        return pd.DataFrame()
    
    for category_path in articles_path.iterdir():
        if category_path.is_dir():
            category = category_path.name
            for article_file in category_path.glob("*.txt"):
                try:
                    with open(article_file, 'r', encoding='utf-8', errors='ignore') as f: 
                        article_content = f.read()
                    summary_file = summaries_path / category / article_file.name
                    if summary_file.exists():
                        with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f: 
                            summary_content = f.read()
                        all_data.append({
                            "filename": article_file.name, 
                            "category": category, 
                            "article": article_content, 
                            "reference_summary": summary_content
                        })
                except Exception as e:
                    print(f"Skipping file {article_file.name} due to error: {e}")
    
    print(f"Loaded {len(all_data)} articles from dataset")
    return pd.DataFrame(all_data)

# ==============================================================================
# --- Main Evaluation Loop ---
# ==============================================================================
def run_svo_xgb_qualitative_benchmark():
    # --- 1. Load Model and Data ---
    print("Loading fine-tuned SVO-XGBoost model...")
    try:
        model = xgb.XGBClassifier()
        model.load_model("svo_xgb_summarizer.json")
        print("SVO-XGBoost model loaded successfully.")
    except (xgb.core.XGBoostError, FileNotFoundError) as e:
        print(f"ERROR: Could not load model 'svo_xgb_summarizer.json': {e}")
        print("Please place your trained model file in the same directory.")
        return

    DATASET_PATH = "./BBC News Summary"
    if not Path(DATASET_PATH).exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return
    
    df = load_bbc_dataset(DATASET_PATH)
    if df.empty:
        print("ERROR: No data loaded from dataset")
        return
    
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    sample_size = min(10, len(test_df))  
    sample_df = test_df.sample(n=50, random_state=42)
    print(f"Running qualitative deep-dive on a sample of {len(sample_df)} articles.")

    # --- 2. Run Evaluation Loop ---
    all_evaluations = []
    successful_evaluations = 0
    
    pbar = tqdm(sample_df.itertuples(), total=len(sample_df), desc="Qualitative Evaluation of SVO-XGBoost")
    for idx, row in enumerate(pbar):
        print(f"\n--- Processing article {idx+1}/{len(sample_df)}: {row.filename} ---")
        
        # Fair Length Control
        try:
            target_len = len(sent_tokenize(row.reference_summary))
        except:
            target_len = 3 # Fallback
        
        print(f"Target summary length: {target_len} sentences")
        
        # Generate the summary
        generated_summary = summarize_with_svo_xgb(row.article, model, num_sents=target_len)
        
        if generated_summary:
            print(f"Generated summary: {generated_summary[:200]}...")
            
            # Get the LLM-as-a-Judge evaluation (reference-free)
            evaluation = llm_as_judge(row.article, generated_summary)
            
            # Check if evaluation was successful (non-zero scores)
            if any(evaluation.get(key, 0) > 0 for key in ['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']):
                successful_evaluations += 1
            
            all_evaluations.append(evaluation)
        else:
            print("Failed to generate summary")
            all_evaluations.append({
                "relevance_score": 0, 
                "faithfulness_score": 0, 
                "coherence_score": 0, 
                "conciseness_score": 0,
                "brief_justification": "Summary generation failed",
                "error": "No summary generated"
            })
        
        time.sleep(2)  

    # --- 3. Print Final Report ---
    eval_df = pd.DataFrame(all_evaluations)
    print("\n\n" + "="*80)
    print("          QUALITATIVE DEEP-DIVE REPORT: Supervised SVO-XGBoost Model          ")
    print("="*80)
    
    print(f"Total samples processed: {len(all_evaluations)}")
    print(f"Successful evaluations: {successful_evaluations}")
    
    if not eval_df.empty and 'relevance_score' in eval_df.columns:
        avg_scores = eval_df[['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']].mean()
        print("\n--- Average Scores Across All Samples (1=Poor, 5=Excellent) ---")
        print(avg_scores.to_frame(name='Average Score').to_string(float_format="{:.2f}".format))
        
        # Show non-zero scores only
        non_zero_mask = (eval_df[['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']] > 0).any(axis=1)
        if non_zero_mask.sum() > 0:
            avg_scores_non_zero = eval_df[non_zero_mask][['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']].mean()
            print(f"\n--- Average Scores for Successful Evaluations ({non_zero_mask.sum()} samples) ---")
            print(avg_scores_non_zero.to_frame(name='Average Score').to_string(float_format="{:.2f}".format))
    else:
        print("\nCould not generate average scores. This might happen if all LLM-as-a-Judge calls failed.")
    
    # Show some example evaluations
    if successful_evaluations > 0:
        print("\n--- Sample Evaluations ---")
        for i, eval_result in enumerate(all_evaluations[:3]):
            if any(eval_result.get(key, 0) > 0 for key in ['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']):
                print(f"Sample {i+1}: {eval_result}")

    print("\n" + "="*80)
    print("Deep-dive complete.")

if __name__ == "__main__":
    run_svo_xgb_qualitative_benchmark()