"""
summarizers/supervised_models.py 
"""

import numpy as np
import spacy
import xgboost as xgb
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from collections import Counter
import re
import pandas as pd 

SPACY_NLP = None
TOKENIZER = None
BERTSUM_MODEL = None
SVO_BERTSUM_MODEL = None
SVO_XGB_MODEL = None
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _load_spacy():
    global SPACY_NLP
    if SPACY_NLP is None:
        print("Loading spaCy model (en_core_web_lg) for supervised models...")
        try:
            SPACY_NLP = spacy.load("en_core_web_lg")
        except OSError:
            print("ERROR: spaCy model 'en_core_web_lg' not found. SVO models will not work.")

# ==============================================================================
# Model 3: SVO-XGBoost 
# ==============================================================================

def _load_svo_xgb_model():
    global SVO_XGB_MODEL
    if SVO_XGB_MODEL is None:
        print("Loading fine-tuned SVO-XGBoost model...")
        try:
            SVO_XGB_MODEL = xgb.XGBClassifier()
            SVO_XGB_MODEL.load_model("svo_xgb_summarizer.json")
        except (xgb.core.XGBoostError, FileNotFoundError):
            print("!!! ERROR: 'svo_xgb_summarizer.json' not found or is invalid.")
            SVO_XGB_MODEL = "error"

def _extract_svo_triples(doc):
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

## MODIFIED ##
def svo_xgb_summarize(text, num_sentences=4):
    _load_spacy()
    _load_svo_xgb_model()

    if SVO_XGB_MODEL == "error" or SPACY_NLP is None:
        return "Model file not found or spaCy model is missing.", {}
    try:
        article_sentences = sent_tokenize(text)
    except Exception:
        return "Could not process text into sentences.", {}
    if not article_sentences: return "Input text is empty.", {}

    article_doc = SPACY_NLP(text)
    article_kg = Counter(_extract_svo_triples(article_doc))
    
    sentence_features = []
    # --- Keep track of original IDs ---
    original_sentence_data = [{'id': i, 'original': s} for i, s in enumerate(article_sentences)]
    
    for i, sentence_text in enumerate(article_sentences):
        sentence_doc = SPACY_NLP(sentence_text)
        sentence_triples = _extract_svo_triples(sentence_doc)
        freqs = [article_kg.get(triple, 0) for triple in sentence_triples]
        features = {'sentence_position': i / len(article_sentences), 'sentence_length': len([token for token in sentence_doc if not token.is_punct]), 'numerical_data_count': len(re.findall(r'\d+', sentence_text)), 'proper_noun_count': len([token for token in sentence_doc if token.pos_ == "PROPN"]), 'num_triples_in_sentence': len(sentence_triples), 'avg_triple_frequency': np.mean(freqs) if freqs else 0, 'max_triple_frequency': np.max(freqs) if freqs else 0}
        sentence_features.append(features)
        
    features_df = pd.DataFrame(sentence_features)
    feature_order = ['sentence_position', 'sentence_length', 'numerical_data_count', 'proper_noun_count', 'num_triples_in_sentence', 'avg_triple_frequency', 'max_triple_frequency']
    features_df = features_df[feature_order]
    predictions = SVO_XGB_MODEL.predict_proba(features_df)[:, 1]
    
    num_to_select = min(num_sentences, len(article_sentences))
    top_indices = sorted(np.argsort(predictions)[-num_to_select:])
    
    summary = " ".join([article_sentences[i] for i in top_indices])
    
    ## Standardize the details dictionary
    details = {
        "processed_sentences": [{'id': s['id'], 'original': s['original'], 'score': p} for s, p in zip(original_sentence_data, predictions)],
        "selected_indices": top_indices,
    }
    
    return summary, details

# ==============================================================================
# Model 4 & 5: BERTSum and SVO-BERTSum 
# ==============================================================================
class BERTSummarizer(torch.nn.Module):
    def __init__(self, model_name=PRE_TRAINED_MODEL_NAME):
        super(BERTSummarizer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(self.bert.config.hidden_size, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    def forward(self, input_ids, attention_mask, cls_indices):
        input_ids = input_ids.squeeze(0); attention_mask = attention_mask.squeeze(0)
        outputs = self.bert(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        last_hidden_state = outputs.last_hidden_state.squeeze(0)
        cls_embeddings = last_hidden_state[cls_indices]
        logits = self.classifier(cls_embeddings)
        return torch.sigmoid(logits)

def _load_bertsum_models():
    global TOKENIZER, BERTSUM_MODEL, SVO_BERTSUM_MODEL
    if TOKENIZER is None: TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    if BERTSUM_MODEL is None:
        print("Loading fine-tuned BERTSum model...")
        try:
            BERTSUM_MODEL = BERTSummarizer().to(DEVICE)
            BERTSUM_MODEL.load_state_dict(torch.load('bertsum_best_model.bin', map_location=DEVICE))
            BERTSUM_MODEL.eval()
        except (FileNotFoundError, RuntimeError):
            print("!!! ERROR: 'bertsum_best_model.bin' not found or is invalid.")
            BERTSUM_MODEL = "error"
    if SVO_BERTSUM_MODEL is None:
        print("Loading fine-tuned SVO-BERTSum model...")
        try:
            SVO_BERTSUM_MODEL = BERTSummarizer().to(DEVICE)
            SVO_BERTSUM_MODEL.load_state_dict(torch.load('svo_bertsum_best_model.bin', map_location=DEVICE))
            SVO_BERTSUM_MODEL.eval()
        except (FileNotFoundError, RuntimeError):
            print("!!! ERROR: 'svo_bertsum_best_model.bin' not found or is invalid.")
            SVO_BERTSUM_MODEL = "error"

def bertsum_summarize_base(text, model_type, probability_threshold=0.5):
    _load_bertsum_models()
    model = BERTSUM_MODEL if model_type == 'standard' else SVO_BERTSUM_MODEL
    if model == "error": return f"Model file for {model_type} BERTSum not found.", {}
    try:
        article_sentences = sent_tokenize(text)
    except Exception:
        return "Could not process text.", {}
    if not article_sentences: return "Input text is empty.", {}

    # --- Keep track of original IDs ---
    original_sentence_data = [{'id': i, 'original': s} for i, s in enumerate(article_sentences)]

    text_for_bert = ""
    if model_type == 'svo_enriched':
        _load_spacy()
        if SPACY_NLP is None: return "spaCy model needed for SVO-BERTSum is missing.", {}
        for sent in article_sentences:
            sent_doc = SPACY_NLP(sent)
            triples = _extract_svo_triples(sent_doc)
            fact_string = ""
            if triples:
                facts = [" ".join(triple) for triple in triples]
                fact_string = " Facts: " + " ; ".join(facts) + "."
            enriched_input = sent + fact_string
            text_for_bert += enriched_input + " [SEP] [CLS] "
    else:
        text_for_bert = " [SEP] [CLS] ".join(article_sentences)
    
    inputs = TOKENIZER.encode_plus(
        text_for_bert, max_length=MAX_LEN, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    cls_indices = (input_ids.squeeze(0) == TOKENIZER.cls_token_id).nonzero().flatten()
    if cls_indices.shape[0] == 0: return "No sentences processed by tokenizer.", {}

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
    
    ## Standardize the details dictionary
    details = {
        "processed_sentences": [{'id': s['id'], 'original': s['original'], 'score': (sentence_scores[i] if i < len(sentence_scores) else 0)} for i, s in enumerate(original_sentence_data)],
        "selected_indices": final_indices,
    }

    return summary, details

def bertsum_summarize(text, title, **kwargs):
    return bertsum_summarize_base(text, model_type='standard')

def svo_bertsum_summarize(text, title, **kwargs):
    return bertsum_summarize_base(text, model_type='svo_enriched')