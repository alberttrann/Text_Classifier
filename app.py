#app.py
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import io
import inspect
import json
import time
from pathlib import Path
import openai
from rouge_score import rouge_scorer
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from bert_score import score

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from summarizers.base_models import textrank_summarize, original_paper_summarize
from summarizers.advanced_extractive import advanced_summarize
from summarizers.supervised_models import svo_xgb_summarize, bertsum_summarize, svo_bertsum_summarize
from summarizers.llm_only import llm_only_summarize

# ==============================================================================
# --- Page Configuration ---
# ==============================================================================
st.set_page_config(
    page_title="Holistic Summarization Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
LOG_DIR = Path("eval_logs")
LOG_DIR.mkdir(exist_ok=True) 
# ==============================================================================
# --- Helper Functions for Log Management ---
# ==============================================================================
def save_log(log_data):
    """Saves a single evaluation log to a timestamped JSON file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = LOG_DIR / f"{timestamp}_{log_data['id']}.json"
    
    # Convert dataframe to JSON string for stable storage
    log_data['results_json'] = log_data['report_df'].to_json(orient='split')
    del log_data['report_df'] 
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)
    st.toast(f"✅ Evaluation log saved to {filename.name}")

def load_logs():
    """Loads all evaluation logs from the log directory, newest first."""
    log_files = sorted(LOG_DIR.glob("*.json"), reverse=True)
    logs = []
    for f in log_files:
        with open(f, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # Reconstruct the DataFrame from the JSON string
                data['report_df'] = pd.read_json(io.StringIO(data['results_json']), orient='split')
                data['filepath'] = f # Keep track of the file path for deletion
                logs.append(data)
            except (json.JSONDecodeError, KeyError):
                print(f"Warning: Could not load or parse log file {f.name}")
                continue
    return logs

def update_note(filepath, new_note):
    """Updates the note in a specific log file."""
    with open(filepath, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data['note'] = new_note
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
    st.toast("Note saved!")

def delete_log(filepath):
    """Deletes a specific log file."""
    try:
        filepath.unlink()
        st.toast(f"✅ Log {filepath.name} deleted.")
    except Exception as e:
        st.error(f"Failed to delete log: {e}")
# ==============================================================================
# --- NEW: Enhanced Evaluation Visualization Functions ---
# ==============================================================================

def calculate_composite_scores(report_df):
    """Calculates multiple composite scoring approaches."""
    normalized_df = report_df.copy()
    positive_metrics = ['ROUGE-L', 'BERTScore-F1', 'NLI Entailment', 'Judge: Relevance', 'Judge: Faithfulness', 'Judge: Coherence', 'Judge: Conciseness']
    negative_metrics = ['NLI Contradiction']
    
    for metric in positive_metrics:
        if metric in normalized_df.columns:
            # Ensure metric is numeric before processing
            normalized_df[metric] = pd.to_numeric(normalized_df[metric], errors='coerce')
            max_val, min_val = normalized_df[metric].max(), normalized_df[metric].min()
            if max_val > min_val:
                normalized_df[f'{metric}_norm'] = (normalized_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{metric}_norm'] = 1.0 if max_val > 0 else 0.0
    
    for metric in negative_metrics:
        if metric in normalized_df.columns:
            normalized_df[metric] = pd.to_numeric(normalized_df[metric], errors='coerce')
            max_val, min_val = normalized_df[metric].max(), normalized_df[metric].min()
            if max_val > min_val:
                normalized_df[f'{metric}_norm'] = 1 - (normalized_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{metric}_norm'] = 1.0

    normalized_df['Quality Score'] = (
        0.30 * normalized_df.get('Judge: Faithfulness_norm', 0) +
        0.25 * normalized_df.get('Judge: Relevance_norm', 0) +
        0.25 * normalized_df.get('Judge: Coherence_norm', 0) +
        0.10 * normalized_df.get('Judge: Conciseness_norm', 0) +
        0.10 * normalized_df.get('NLI Contradiction_norm', 0)
    )
    normalized_df['Similarity Score'] = (
        0.50 * normalized_df.get('BERTScore-F1_norm', 0) +
        0.30 * normalized_df.get('ROUGE-L_norm', 0) +
        0.20 * normalized_df.get('NLI Entailment_norm', 0)
    )
    
    available_metrics = [col for col in normalized_df.columns if col.endswith('_norm')]
    if available_metrics:
        normalized_df['Balanced Score'] = normalized_df[available_metrics].mean(axis=1)
    
    return normalized_df

def create_radar_chart(report_df):
    """Creates a radar chart comparing all models across key metrics."""
    if not PLOTLY_AVAILABLE: return None
    radar_metrics = ['Judge: Relevance', 'Judge: Faithfulness', 'Judge: Coherence', 'ROUGE-L', 'BERTScore-F1', 'NLI Entailment']
    available_metrics = [m for m in radar_metrics if m in report_df.columns]
    if len(available_metrics) < 3: return None
    
    fig = go.Figure()
    for idx, row in report_df.iterrows():
        model_name = row['Model']
        values = []
        for metric in available_metrics:
            val = pd.to_numeric(row.get(metric, 0), errors='coerce')
            if "Judge:" in metric:
                normalized_val = val
            else:
                normalized_val = val * 5
            values.append(normalized_val)
        
        values.append(values[0])
        theta_labels = available_metrics + [available_metrics[0]]
        fig.add_trace(go.Scatterpolar(r=values, theta=theta_labels, fill='toself', name=model_name))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), title="Model Performance Comparison (Radar Chart)")
    return fig

def create_performance_heatmap(report_df):
    """Creates a heatmap showing model performance across all metrics."""
    if not PLOTLY_AVAILABLE: return None
    metrics_cols = [col for col in report_df.columns if col != 'Model']
    heatmap_data = report_df[metrics_cols].copy()
    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
        if col != 'NLI Contradiction':
            max_val, min_val = heatmap_data[col].max(), heatmap_data[col].min()
            if max_val > min_val: heatmap_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
        else:
            max_val, min_val = heatmap_data[col].max(), heatmap_data[col].min()
            if max_val > min_val: heatmap_data[col] = 1 - (heatmap_data[col] - min_val) / (max_val - min_val)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=heatmap_data.columns, y=report_df['Model'].values,
        colorscale='RdYlGn', text=[[f'{report_df.iloc[i][col]:.3f}' for col in heatmap_data.columns] for i in range(len(report_df))],
        texttemplate="%{text}", textfont={"size": 10}
    ))
    fig.update_layout(title="Model Performance Heatmap (Normalized: Green=Best)", yaxis_autorange='reversed')
    return fig

def display_enhanced_evaluation_results(report_df):
    """Main function to display all enhanced visualizations."""
    st.markdown("---")
    st.subheader("📊 Enhanced Benchmark Visualization")
    if not PLOTLY_AVAILABLE:
        st.error("Plotly library not found. Please install it (`pip install plotly`) to view enhanced visualizations.")
        st.dataframe(report_df)
        return

    try:
        if report_df.empty or 'Model' not in report_df.columns:
            st.error("Invalid report data."); return
            
        scored_df = calculate_composite_scores(report_df)
        composite_cols = ['Quality Score', 'Similarity Score', 'Balanced Score']
        available_composite = [col for col in composite_cols if col in scored_df.columns]
        
        viz_tabs = st.tabs(["📈 Overview", "🎯 Radar Chart", "🔥 Heatmap"])
        
        with viz_tabs[0]:
            st.markdown("#### Composite Scoring Summary")
            cols = st.columns(len(available_composite))
            for i, score_type in enumerate(available_composite):
                with cols[i]:
                    best_model = scored_df.loc[scored_df[score_type].idxmax()]
                    st.metric(label=f"🏆 Best in {score_type.split()[0]}", value=best_model['Model'], delta=f"{best_model[score_type]:.3f}")
            display_cols = ['Model'] + available_composite + [col for col in report_df.columns if col != 'Model']
            st.dataframe(scored_df[display_cols].sort_values(available_composite[0], ascending=False), use_container_width=True)
        
        with viz_tabs[1]:
            st.plotly_chart(create_radar_chart(report_df), use_container_width=True)
        
        with viz_tabs[2]:
            st.plotly_chart(create_performance_heatmap(report_df), use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")
        st.dataframe(report_df)
# ==============================================================================
# --- Central Evaluation Engine ---
# ==============================================================================
@st.cache_resource
def get_evaluation_runner():
    """
    A cached singleton class to hold all heavy evaluation models so they are
    loaded only once.
    """
    class EvaluationRunner:
        def __init__(self):
            print("Initializing EvaluationRunner (loading models)...")
            # Cloud LLM for Judge
            self.cloud_client = None
            try:
                API_KEY = ""
                self.CLOUD_MODEL_NAME = "gpt-oss-120b"  
                self.cloud_client = openai.OpenAI(api_key=API_KEY, base_url="https://mkp-api.fptcloud.com")
                print("Cloud LLM client for judging configured.")
            except Exception as e:
                print(f"Failed to configure OpenAI client for judging: {e}")

            # ROUGE Scorer
            self.rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # NLI Scorer Models
            self.nli_pipeline = pipeline("text-classification", model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', device=-1)
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("EvaluationRunner Initialized.")

        def run_nli_score(self, article_text, summary_text):
            if not article_text or not summary_text: return 0.0, 0.0
            try:
                article_sentences = sent_tokenize(article_text)
                summary_sentences = sent_tokenize(summary_text)
                if not article_sentences or not summary_sentences: return 0.0, 0.0
            except Exception: return 0.0, 0.0
            article_embeddings = self.sbert_model.encode(article_sentences, convert_to_tensor=True, show_progress_bar=False)
            entailment_scores, contradiction_scores = [], []
            for summary_sent in summary_sentences:
                summary_sent_embedding = self.sbert_model.encode(summary_sent, convert_to_tensor=True, show_progress_bar=False)
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
            return np.mean(entailment_scores) if entailment_scores else 0.0, np.max(contradiction_scores) if contradiction_scores else 0.0
        def run_llm_as_judge(self, original_article, candidate_summary):
            """
            Uses a cloud LLM to score a candidate summary based ONLY on the original article.
            """
            if not self.cloud_client:
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
                response = self.cloud_client.chat.completions.create(
                    model=self.CLOUD_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024,
                )
                
                if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                    print("LLM judge error: Received an empty or invalid response from the API.")
                    default_response["error"] = "API returned empty content."
                    return default_response
                
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
                return final_evaluation

            except Exception as e:
                print(f"LLM judge error: {e}")  
                default_response["error"] = str(e)
                return default_response

        def run_full_evaluation(self, summaries_dict, article, reference):
            report_data = []
            for model_name, summary_text in summaries_dict.items():
                if not summary_text: continue
                
                # ROUGE
                rouge_scores = self.rouge_eval.score(reference, summary_text)
                # BERTScore (run as a batch later for efficiency)
                # NLI
                nli_entail, nli_contra = self.run_nli_score(article, summary_text)
                # LLM Judge
                judge_scores = self.run_llm_as_judge(article, summary_text)
                
                report_data.append({
                    "Model": model_name,
                    "ROUGE-L": rouge_scores['rougeL'].fmeasure,
                    "NLI Entailment": nli_entail,
                    "NLI Contradiction": nli_contra,
                    "Judge: Relevance": judge_scores['relevance_score'],
                    "Judge: Faithfulness": judge_scores['faithfulness_score'],
                    "Judge: Coherence": judge_scores['coherence_score'],
                    "Judge: Conciseness": judge_scores['conciseness_score']
                })
            # Batch process BERTScore for all non-empty summaries
            model_names = [d["Model"] for d in report_data]
            candidates = [summaries_dict[name] for name in model_names]
            _, _, bert_f1 = score(candidates, [reference]*len(candidates), lang="en", verbose=False)  
            
            for i, data in enumerate(report_data):
                data["BERTScore-F1"] = bert_f1[i].item()
                
            return pd.DataFrame(report_data)
            return pd.DataFrame(report_data)

    return EvaluationRunner()
# ==============================================================================
# --- Helper & Visualization Functions ---
# ==============================================================================
def draw_graph_visualization(details):
    """Creates a graph visualization for TextRank and Original Paper models."""
    graph = details.get("graph")
    if not graph:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(graph.nodes) > 0:
        pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
        selected_indices = details.get("selected_indices", [])
        node_colors = ['#FFD700' if i in selected_indices else '#1f78b4' for i in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=500, alpha=0.9)
        
        if graph.edges(data=True):
            weights = [d.get('weight', 0.5) for u, v, d in graph.edges(data=True)]
            max_weight = max(weights) if weights else 1.0
            edge_widths = [2.5 * (w / max_weight) for w in weights]
            nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, alpha=0.5, edge_color='gray')

        labels = {i: f"S{i+1}" for i in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=10, font_weight='bold')

    ax.set_title("Sentence Similarity Graph (Selected Sentences Highlighted)", fontsize=16)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    return buf

def display_rerank_and_hybrid_info(details, is_hybrid):
    """
    A unified visualization for all models that use the Generate-and-Re-rank pipeline.
    """
    sentence_lookup = {s.get('id'): s for s in details.get("processed_sentences", [])}

    st.subheader("🕵️‍♂️ Behind the Scenes: The Two-Stage Process")

    # --- Stage 1: Extractive Content Selection ---
    st.markdown("#### Stage 1: Extractive Content Selection")
    st.info("The model runs an internal contest to find the single best **extractive** summary (the 'fact sheet'). It generates multiple candidates with different 'personalities', and a 'Re-ranker' judges them to select the winner.")
    
    rerank_details = details.get("rerank_details", [])
    if not rerank_details:
        st.warning("Re-ranking details are not available."); return

    st.markdown("**Candidate Evaluation:**")
    df_data = []
    best_score = max((cand.get('score', -1) for cand in rerank_details), default=-1)
    for cand in rerank_details:
        df_data.append({
            "🏆 Winner": "✅" if cand.get('score') == best_score else "",
            "Personality": cand.get("name", "N/A"),
            "Structure Score": f"{cand.get('structure', 0):.2f}",
            "Balance Score": f"{cand.get('balance', 0):.2f}",
            "Coherence Score": f"{cand.get('coherence', 0):.2f}",
            "**Final Score**": f"**{cand.get('score', 0):.3f}**"
        })
    st.dataframe(pd.DataFrame(df_data))
    
    ## Expander to show the content of each candidate
    with st.expander("📄 View All Competing Candidate Summaries"):
        for cand in rerank_details:
            st.markdown(f"--- \n**Candidate: {cand.get('name')} (Score: {cand.get('score', 0):.3f})**")
            # Use the full sentence_lookup that has ALL original sentences
            summary_text = " ".join([sentence_lookup[i]['original'] for i in cand.get('indices', []) if i in sentence_lookup])
            st.write(summary_text if summary_text else "_(This candidate was empty)_")

    # --- Stage 2: Abstractive Polishing (only for hybrid models) ---
    if is_hybrid:
        st.markdown("---")
        st.markdown("#### Stage 2: Abstractive Polishing")
        st.info("Next, the winning 'fact sheet' is sent to an LLM with a specific prompt, asking it to rewrite the facts into a single, fluent paragraph.")
        
        polish_details = details.get("polish_details", {})
        
        st.markdown("**Winning 'Fact Sheet' Sent to LLM:**")
        st.success(details.get("extractive_summary", "Not available."))
        
        ## Correctly display the prompt or fact-checking log
        # Check the model name from the session state to decide which log to show
        if "fact-checked" in st.session_state.get('current_model_name', '').lower():
            display_fact_checking_log(polish_details)
        else:
            st.markdown("**Prompt Sent to LLM:**")
            prompt = polish_details.get("prompt_sent_to_llm", "Prompt was not captured by the backend.")
            st.code(prompt, language='text')

def display_fact_checking_log(details):
    """A helper to display the detailed fact-checking log for the winning model."""
    initial_summary = details.get("initial_summary")
    log = details.get("fact_checking_log", [])
    if not initial_summary or not log:
        st.warning("Fact-checking details are not available for this run."); return
    st.markdown("**A. Initial Polished Summary (from LLM):**"); st.info(initial_summary)
    st.markdown("**B. Verification Process:**")
    for item in log:
        status, original_sent = item.get("status"), item.get("original")
        if status == "PASSED": st.markdown(f"✅ **PASSED:** `{original_sent}`")
        elif status == "REWRITTEN": st.markdown(f"❌ **FAILED & REWRITTEN:**\n   - **Original:** `{original_sent}`\n   - **Corrected:** `{item.get('corrected')}`")
        elif status == "DISCARDED": st.markdown(f"🗑️ **FAILED & DISCARDED:** `{original_sent}`")
        elif status == "ERROR": st.error(f"**ERROR:** {original_sent}")

def display_probability_scores(details):
    """Displays the probability scores from supervised models."""
    st.subheader("🎯 Sentence Probability Scores"); st.write("Sentences with higher probabilities are selected.")
    sentences_data = details.get("processed_sentences", []); selected_indices = details.get("selected_indices", [])
    if not sentences_data: return
    score_data = [{"Selected": "✅" if s.get('id') in selected_indices else "", "Probability": f"{s.get('score', 0.0):.3f}", "Sentence": s.get('original', '')} for s in sentences_data]
    st.dataframe(score_data)

# ==============================================================================
# --- Main Application UI ---
# ==============================================================================

st.title("📚 Holistic Text Summarization Dashboard")
st.markdown("An interactive, on-demand dashboard to compare and evaluate nine different summarization architectures.")

# --- Session State & Sidebar ---
if 'doc_title' not in st.session_state: st.session_state.doc_title = ""
if 'doc_text' not in st.session_state: st.session_state.doc_text = ""
if 'results' not in st.session_state: st.session_state.results = {}
if 'evaluation_report' not in st.session_state: st.session_state.evaluation_report = None

st.sidebar.title("Controls")
def update_text():
    st.session_state.results = {}
    st.session_state.evaluation_report = None
st.sidebar.header("1. Input Document")
st.session_state.doc_title = st.sidebar.text_input("Document Title", st.session_state.doc_title, on_change=update_text, placeholder="e.g., Time travel")
st.session_state.doc_text = st.sidebar.text_area("Document Text", st.session_state.doc_text, height=250, on_change=update_text, placeholder="Paste your article text here...")
st.sidebar.header("2. Model Parameters")
detail_level = st.sidebar.select_slider("Select Detail Level (Advanced/Hybrid/LLM)", options=['Balanced', 'Detailed'], value='Balanced')
num_sents_classic = st.sidebar.number_input("Number of Sentences (Classic/Supervised)", min_value=2, max_value=20, value=4)

# --- Main Content Area with Tabs ---
st.header("Summarization Model Outputs")
st.markdown("Click the 'Generate' button within any tab to run that specific model.")

tab_definitions = {
    "Original Paper": {"func": original_paper_summarize},
    "TextRank": {"func": textrank_summarize},
    "SVO-XGBoost": {"func": svo_xgb_summarize},
    "BERTSum": {"func": bertsum_summarize},
    "SVO-BERTSum": {"func": svo_bertsum_summarize},
    "Adv. Extractive": {"func": advanced_summarize},
    "Hybrid (Original)": {"func": advanced_summarize},
    "LLM-Only": {"func": llm_only_summarize},
    "Hybrid (Fact-Checked)": {"func": advanced_summarize}
}

tab_names = list(tab_definitions.keys()) + ["📊 Full Evaluation", "📝 Eval Log"]
tabs = st.tabs(tab_names)
for i, (model_name, config) in enumerate(tab_definitions.items()):
    with tabs[i]:
        st.subheader(f"Model: {model_name}")
        
        # Display results if they are in the session state
        if model_name in st.session_state.results:
            result = st.session_state.results[model_name]
            summary, details = result['summary'], result['details']
            st.markdown("**Generated Summary:**")
            st.success(summary)
            
            with st.expander("🕵️‍♂️ View Behind the Scenes"):
                if "adv" in model_name.lower() or "hybrid" in model_name.lower():
                    is_hybrid_flag = "hybrid" in model_name.lower()
                    display_rerank_and_hybrid_info(details, is_hybrid=is_hybrid_flag)
                elif "bert" in model_name.lower() or "xgb" in model_name.lower():
                    display_probability_scores(details)
                elif "rank" in model_name.lower() or "paper" in model_name.lower():
                    graph_img = draw_graph_visualization(details)
                    if graph_img: st.image(graph_img)
                elif "llm-only" in model_name.lower():
                     st.code(details.get("prompt_sent_to_llm", "Prompt not available."), language='text')

        # The "Generate" button is inside each tab
        if st.button(f"Generate with {model_name}", key=f"btn_{model_name}"):
            if not st.session_state.doc_text or st.session_state.doc_text == "Paste your article text here...":
                st.error("Please paste text in the sidebar first.")
            else:
                with st.spinner(f"Running {model_name}... This can take over a minute for complex models."):
                    st.session_state['current_model_name'] = model_name
                    func = config['func']
                    
                    # Build the arguments dictionary dynamically and safely.
                    all_possible_params = {
                        "text": st.session_state.doc_text,
                        "title": st.session_state.doc_title,
                        "num_sentences": num_sents_classic,
                        "detail_level": detail_level,
                        "enable_hybrid": "Hybrid" in model_name,
                        "enable_fact_checking": "Fact-Checked" in model_name,
                    }
                    func_signature = inspect.signature(func)
                    accepted_args = func_signature.parameters.keys()
                    kwargs_to_pass = {k: v for k, v in all_possible_params.items() if k in accepted_args}
                    
                    summary, details = func(**kwargs_to_pass)
                    
                    st.session_state.results[model_name] = {'summary': summary, 'details': details}
                    st.rerun()

# --- Initial Page Load Message ---
if not st.session_state.results:
    st.info("Enter a title and text in the sidebar, then click 'Generate' within a tab to begin.")

# ==============================================================================
# --- Logic for the Full Evaluation Tab ---
# ==============================================================================
with tabs[-2]:
    st.subheader("📊 Grand Unified Benchmark")
    st.markdown("Click the button to generate summaries with all models under a **fair, length-controlled setting** and compare them across all metrics.")
    st.warning("**Note:** This is a computationally intensive process that will re-run all 9 models. It may take several minutes.")

    if st.button("🚀 Run Full Evaluation", type="primary"):
        if not st.session_state.doc_text:
            st.error("Please paste text in the sidebar first.")
        else:
            all_model_names = list(tab_definitions.keys())
            summaries_for_eval = {}
            
            # --- 1. NEW: LLM-Driven Fair Length Control ---
            with st.spinner("Step 1: Asking Cloud LLM to determine the ideal summary length..."):
                try:
                    # A prompt designed to make the LLM perform extractive summarization
                    length_determination_prompt = (
                        "You are a summarization expert. Your task is to perform a purely extractive summary of the following article. "
                        "Read the entire article, identify the most important sentences that capture the core message, and list them verbatim. "
                        "At the very end, state the total count of sentences you selected in the format: 'SENTENCE_COUNT: <number>'.\n\n"
                        "--- ARTICLE ---\n"
                        "{text}\n\n"
                        "--- EXTRACTED SENTENCES AND COUNT ---"
                    ).format(text=st.session_state.doc_text)
                    
                    # Get the singleton instance of the runner to access the client
                    eval_runner = get_evaluation_runner()
                    
                    response = eval_runner.cloud_client.chat.completions.create(
                        model=eval_runner.CLOUD_MODEL_NAME,
                        messages=[{"role": "user", "content": length_determination_prompt}],
                        temperature=0.0 # Zero temperature for deterministic, factual tasks
                    )
                    llm_response_text = response.choices[0].message.content.strip()
                    
                    # Extract the number from the "SENTENCE_COUNT: X" line
                    match = re.search(r"SENTENCE_COUNT:\s*(\d+)", llm_response_text)
                    if match:
                        target_len = int(match.group(1))
                        # Use the LLM's extractive output as the pseudo-reference
                        pseudo_reference = re.sub(r"SENTENCE_COUNT:\s*(\d+)", "", llm_response_text).strip()
                        st.info(f"Cloud LLM determined a fair summary length: **{target_len} sentences**.")
                    else:
                        raise ValueError("LLM did not return the sentence count in the expected format.")

                except Exception as e:
                    st.warning(f"Cloud LLM length determination failed ({e}). Falling back to TextRank-based length.")
                    pseudo_reference = textrank_summarize(st.session_state.doc_text)[0]
                    target_len = len(sent_tokenize(pseudo_reference))
                    st.info(f"Fallback length established: **{target_len} sentences**.")
            
            # --- 2. Generate ALL Summaries with the SAME Target Length ---
            progress_bar = st.progress(0, text="Generating summaries for all models...")
            all_model_names = list(tab_definitions.keys())
            
            for i, model_name in enumerate(all_model_names):
                progress_text = f"Generating summary... ({i+1}/{len(all_model_names)}: {model_name})"
                progress_bar.progress((i+1) / len(all_model_names), text=progress_text)
                
                config = tab_definitions[model_name]
                func = config['func']
                
                # Build the arguments for THIS model, enforcing the target length
                kwargs_to_pass = {"text": st.session_state.doc_text, "title": st.session_state.doc_title}
                len_param_name = config.get("fair_length_param_name")

                # Override the length parameter with our fair target length
                if len_param_name:
                    kwargs_to_pass[len_param_name] = target_len
                
                # Only pass accepted arguments to the function
                func_signature = inspect.signature(func)
                final_kwargs = {k: v for k, v in kwargs_to_pass.items() if k in func_signature.parameters}

                summary, _ = func(**final_kwargs)
                summaries_for_eval[model_name] = summary

            progress_bar.empty()

            # --- 3. Run the Full Evaluation ---
            with st.spinner("All summaries generated fairly. Running full evaluation..."):
                eval_runner = get_evaluation_runner()
                
                # For ROUGE/BERTScore, we need a reference. The pseudo-reference is fine for this on-demand task.
                pseudo_reference = pseudo_reference

                report_df = eval_runner.run_full_evaluation(summaries_for_eval, st.session_state.doc_text, pseudo_reference)
                st.session_state.evaluation_report = report_df
            
            log_id = re.sub(r'\W+', '', st.session_state.doc_title.lower())[:20]
            log_data = {
                "id": log_id,
                "title": st.session_state.doc_title,
                "article": st.session_state.doc_text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "", 
                "report_df": report_df 
            }
            save_log(log_data)
            
            st.rerun()

    # --- 4. Display the final report ---
    if st.session_state.evaluation_report is not None:
        st.markdown("---")
        st.subheader("Benchmark Results")
        
        report_df = st.session_state.evaluation_report
        # Calculate an overall score for ranking
        report_df['Overall Score'] = (
            report_df['Judge: Relevance'] +
            report_df['Judge: Faithfulness'] +
            report_df['Judge: Coherence']
        )
        report_df = report_df.sort_values(by="Overall Score", ascending=False)
        
        st.dataframe(
            report_df,
            column_config={
                "ROUGE-L": st.column_config.ProgressColumn("ROUGE-L", format="%.3f", min_value=0, max_value=1),
                "BERTScore-F1": st.column_config.ProgressColumn("BERTScore-F1", format="%.3f", min_value=0, max_value=1),
                "NLI Entailment": st.column_config.ProgressColumn("NLI Entailment", format="%.3f", min_value=0, max_value=1),
            }
        )

with tabs[-1]: 
    st.subheader("📝 Evaluation Log")
    st.markdown("Review, annotate, and manage your past benchmark runs.")
    
    all_logs = load_logs()
    
    if not all_logs:
        st.info("No evaluation logs found. Run a 'Full Evaluation' in the previous tab to save a new log.")
    else:
        st.markdown(f"Found **{len(all_logs)}** saved evaluation(s).")
        
        # Display each log in an expander
        for i, log in enumerate(all_logs):
            exp_title = f"📄 **{log.get('title', 'Untitled')}** (Saved: {log.get('timestamp', 'N/A')})"
            with st.expander(exp_title):
                
                st.markdown("---")
                st.markdown("#### Benchmark Results")
                
                # Display the dataframe from the log
                report_df = log['report_df']
                report_df['Overall Score'] = (report_df['Judge: Relevance'] + report_df['Judge: Faithfulness'] + report_df['Judge: Coherence'])
                report_df = report_df.sort_values(by="Overall Score", ascending=False)
                st.dataframe(
                    report_df,
                    column_config={
                        "ROUGE-L": st.column_config.ProgressColumn("ROUGE-L", format="%.3f", min_value=0, max_value=1),
                        "BERTScore-F1": st.column_config.ProgressColumn("BERTScore-F1", format="%.3f", min_value=0, max_value=1),
                        "NLI Entailment": st.column_config.ProgressColumn("NLI Entailment", format="%.3f", min_value=0, max_value=1),
                    }
                )
                if st.button("📊 Visualize this Result", key=f"viz_{log['filepath'].name}"):
                    # Store which log to visualize in the session state
                    st.session_state.log_to_visualize = log['filepath'].name
                
                if st.session_state.get('log_to_visualize') == log['filepath'].name:
                    display_enhanced_evaluation_results(log['report_df'])
                
                st.markdown("---")
                st.markdown("#### Notes")
                
                # Note taking functionality
                note_key = f"note_{log['filepath'].name}"
                current_note = log.get('note', '')
                new_note = st.text_area("Add or edit your notes for this run:", value=current_note, key=note_key, height=100)
                
                if st.button("Save Note", key=f"save_{log['filepath'].name}"):
                    update_note(log['filepath'], new_note)
                    st.rerun() 
                
                st.markdown("---")
                with st.expander("View Original Article Text"):
                    st.text(log.get('article', 'Article text not found in log.'))
                    
                st.markdown("---")
                st.markdown("#### Manage Log")
                delete_key = f"delete_{log['filepath'].name}"
                confirm_key = f"confirm_{log['filepath'].name}"

                if st.button("Delete this log", key=delete_key, type="secondary"):
                    st.session_state[confirm_key] = True
                
                if st.session_state.get(confirm_key, False):
                    st.warning(f"**Are you sure you want to permanently delete the log for '{log.get('title')}'?**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes, Delete Permanently", key=f"final_delete_{log['filepath'].name}", type="primary"):
                            delete_log(log['filepath'])
                            st.session_state[confirm_key] = False 
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key=f"cancel_delete_{log['filepath'].name}"):
                            st.session_state[confirm_key] = False 
                            st.rerun()