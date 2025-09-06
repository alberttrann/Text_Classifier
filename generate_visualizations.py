import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

def clean_and_prepare_data(table_string):
    """
    Parses the markdown table string, cleans it, and returns a pandas DataFrame.
    """
    # Remove the markdown bold characters '**'
    cleaned_string = re.sub(r'\*\*', '', table_string)
    
    # Use StringIO to read the string data into pandas
    data = io.StringIO(cleaned_string)
    df = pd.read_csv(data, sep='|', skipinitialspace=True)
    
    # --- Data Cleaning ---
    # Drop the first and last columns which are empty due to the table format
    df = df.iloc[:, 1:-1]
    # Drop the header separator line
    df = df.drop(0)
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    # Strip whitespace from data cells
    for col in df.columns:
        df[col] = df[col].str.strip()
        
    # Set the model name as the index
    df = df.rename(columns={'Model / Configuration': 'Model'})
    df = df.set_index('Model')
    
    # Filter out category rows (like 'Supervised Extractive Models')
    category_rows = [
        'Supervised Extractive Models', 
        'Unsupervised Extractive Models', 
        'Abstractive & Hybrid Models'
    ]
    df = df[~df.index.isin(category_rows)]
    
    # Convert all metric columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Rename columns for easier access
    df.columns = [
        'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 
        'NLI Entailment', 'NLI Contradiction', 'Judge: Relevance', 
        'Judge: Faithfulness', 'Judge: Coherence', 'Judge: Conciseness', 
        'Judge: Overall Avg'
    ]
    
    return df

def plot_rouge_scores(df):
    """Plots ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    rouge_df = df[['ROUGE-1', 'ROUGE-2', 'ROUGE-L']].copy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = rouge_df.plot(kind='bar', figsize=(14, 8), colormap='viridis')
    
    plt.title('Comparison of ROUGE Scores Across Models', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='ROUGE Metric')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=2)
        
    plt.savefig("rouge_scores_comparison.png", dpi=300)
    print("Saved plot to rouge_scores_comparison.png")

def plot_semantic_scores(df):
    """Plots BERTScore and NLI scores."""
    semantic_df = df[['BERTScore-F1', 'NLI Entailment', 'NLI Contradiction']].copy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = semantic_df.plot(kind='bar', figsize=(14, 8), colormap='plasma')
    
    plt.title('Semantic & Factual Consistency Scores', fontsize=16, fontweight='bold')
    plt.ylabel('Score (Higher is better for F1/Entailment, Lower for Contradiction)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.ylim(0, 1.1)
    plt.tight_layout()

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=2)

    plt.savefig("semantic_scores_comparison.png", dpi=300)
    print("Saved plot to semantic_scores_comparison.png")

def plot_judge_scores(df):
    """Plots the four individual LLM-as-a-Judge scores."""
    judge_df = df[['Judge: Relevance', 'Judge: Faithfulness', 'Judge: Coherence', 'Judge: Conciseness']].copy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = judge_df.plot(kind='bar', figsize=(14, 8), colormap='magma')
    
    plt.title('LLM-as-a-Judge Qualitative Evaluation (1-5 Scale)', fontsize=16, fontweight='bold')
    plt.ylabel('Average Score (1=Poor, 5=Excellent)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Evaluation Criteria')
    plt.ylim(0, 5.5)
    plt.tight_layout()

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=2)

    plt.savefig("judge_scores_comparison.png", dpi=300)
    print("Saved plot to judge_scores_comparison.png")

def plot_overall_judge_ranking(df):
    """Plots a horizontal bar chart for the overall average judge score."""
    overall_df = df[['Judge: Overall Avg']].copy().sort_values(by='Judge: Overall Avg', ascending=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = overall_df.plot(kind='barh', figsize=(12, 8), colormap='cividis')
    
    plt.title('Model Ranking by Overall Average Judge Score', fontsize=16, fontweight='bold')
    plt.xlabel('Average Score (1-5)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.legend().set_visible(False)
    plt.xlim(0, 5)
    
    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10, padding=5)
        
    plt.tight_layout()
    plt.savefig("overall_judge_ranking.png", dpi=300)
    print("Saved plot to overall_judge_ranking.png")


if __name__ == '__main__':
    # Paste the markdown table here
    markdown_table = """
| Model / Configuration                | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 | NLI Entailment (↑) | NLI Contradiction (↓) | Judge: Relevance (1-5) | Judge: Faithfulness (1-5) | Judge: Coherence (1-5) | Judge: Conciseness (1-5) | **Judge: Overall Avg** |
| :----------------------------------- | :------ | :------ | :------ | :----------- | :----------------- | :-------------------- | :--------------------- | :------------------------ | :--------------------- | :----------------------- | :--------------------- |
| **Supervised Extractive Models**     |         |         |         |              |                    |                       |                        |                           |                        |                          |                        |
| `BERTSum (Fine-tuned)`               | **0.7634**  | **0.7035**  | **0.4993**  | **0.9308**   | **0.9843**         | 0.0095                | 2.86                   | 4.90                      | 4.48                   | 4.24                     | 4.12                   |
| `SVO-BERTSum (Fine-tuned)`           | 0.7512  | 0.6883  | 0.4845  | 0.9282       | **0.9843**         | **0.0095**                | 2.80                   | **4.92**                      | 4.42                   | 4.22                     | 4.09                   |
| `SVO-XGBoost (Supervised)`           | 0.3938  | 0.3235  | 0.3126  | 0.8849       | **0.9846**         | **0.0037**                | 2.88                   | 4.84                      | 4.68                   | 4.12                     | 4.13                   |
| **Unsupervised Extractive Models**   |         |         |         |              |                    |                       |                        |                           |                        |                          |                        |
| `TextRank_Baseline`                  | 0.4191  | 0.3878  | 0.3429  | 0.9019       | **0.9845**         | **0.0053**                | 1.48                   | 4.90                      | 4.34                   | 4.74                     | 3.87                   |
| `Advanced_Extractive (Balanced)`     | 0.3597  | 0.2823  | 0.2899  | 0.8779       | 0.9782             | **0.0053**                | 1.58                   | **4.96**                      | 4.34                   | **4.82**                     | 3.93                   |
| `Original_Paper_Method`              | 0.3892  | 0.3102  | 0.3093  | 0.8806       | 0.9827             | **0.0053**                | 1.84                   | 4.92                      | 4.36                   | 4.72                     | 3.96                   |
| **Abstractive & Hybrid Models**      |         |         |         |              |                    |                       |                        |                           |                        |                          |                        |
| `Hybrid (Faithfulness-Guaranteed)`   | 0.3534  | 0.2567  | 0.2728  | 0.8793       | **0.9833**         | **0.0000**                | 2.26                   | 4.88                      | 4.60                   | 4.10                     | 3.96                   |
| `Hybrid (Original)`                  | 0.3172  | 0.1078  | 0.1785  | 0.8609       | 0.4410             | 0.0805                | **3.18**                   | 4.92                      | **4.90**                   | **4.90**                     | **4.48**                   |
| `LLM_Only (Unsafe)`                  | 0.3016  | 0.1027  | 0.1686  | 0.8610       | 0.1393             | 0.0703                | 1.00                   | 1.02                      | 2.76                   | 2.94                     | 1.93                   |
    """
    
    # Prepare data
    results_df = clean_and_prepare_data(markdown_table)
    
    # Generate plots
    plot_rouge_scores(results_df)
    plot_semantic_scores(results_df)
    plot_judge_scores(results_df)
    plot_overall_judge_ranking(results_df)
    
    print("\nAll visualizations have been generated and saved as PNG files.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

def clean_and_prepare_data(table_string):
    """
    Parses the markdown table string, cleans it, and returns a pandas DataFrame.
    """
    # Remove the markdown bold characters '**'
    cleaned_string = re.sub(r'\*\*', '', table_string)
    
    # Use StringIO to read the string data into pandas
    data = io.StringIO(cleaned_string)
    df = pd.read_csv(data, sep='|', skipinitialspace=True)
    
    # --- Data Cleaning ---
    # Drop the first and last columns which are empty due to the table format
    df = df.iloc[:, 1:-1]
    # Drop the header separator line
    df = df.drop(0)
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    # Strip whitespace from data cells
    for col in df.columns:
        df[col] = df[col].str.strip()
        
    # Set the model name as the index
    df = df.rename(columns={'Model / Configuration': 'Model'})
    df = df.set_index('Model')
    
    # Filter out category rows (like 'Supervised Extractive Models')
    category_rows = [
        'Supervised Extractive Models', 
        'Unsupervised Extractive Models', 
        'Abstractive & Hybrid Models'
    ]
    df = df[~df.index.isin(category_rows)]
    
    # Convert all metric columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Rename columns for easier access
    df.columns = [
        'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 
        'NLI Entailment', 'NLI Contradiction', 'Judge: Relevance', 
        'Judge: Faithfulness', 'Judge: Coherence', 'Judge: Conciseness', 
        'Judge: Overall Avg'
    ]
    
    return df

def plot_rouge_scores(df):
    """Plots ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    rouge_df = df[['ROUGE-1', 'ROUGE-2', 'ROUGE-L']].copy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = rouge_df.plot(kind='bar', figsize=(14, 8), colormap='viridis')
    
    plt.title('Comparison of ROUGE Scores Across Models', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='ROUGE Metric')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=2)
        
    plt.savefig("rouge_scores_comparison.png", dpi=300)
    print("Saved plot to rouge_scores_comparison.png")

def plot_semantic_scores(df):
    """Plots BERTScore and NLI scores."""
    semantic_df = df[['BERTScore-F1', 'NLI Entailment', 'NLI Contradiction']].copy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = semantic_df.plot(kind='bar', figsize=(14, 8), colormap='plasma')
    
    plt.title('Semantic & Factual Consistency Scores', fontsize=16, fontweight='bold')
    plt.ylabel('Score (Higher is better for F1/Entailment, Lower for Contradiction)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.ylim(0, 1.1)
    plt.tight_layout()

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=2)

    plt.savefig("semantic_scores_comparison.png", dpi=300)
    print("Saved plot to semantic_scores_comparison.png")

def plot_judge_scores(df):
    """Plots the four individual LLM-as-a-Judge scores."""
    judge_df = df[['Judge: Relevance', 'Judge: Faithfulness', 'Judge: Coherence', 'Judge: Conciseness']].copy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = judge_df.plot(kind='bar', figsize=(14, 8), colormap='magma')
    
    plt.title('LLM-as-a-Judge Qualitative Evaluation (1-5 Scale)', fontsize=16, fontweight='bold')
    plt.ylabel('Average Score (1=Poor, 5=Excellent)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Evaluation Criteria')
    plt.ylim(0, 5.5)
    plt.tight_layout()

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, padding=2)

    plt.savefig("judge_scores_comparison.png", dpi=300)
    print("Saved plot to judge_scores_comparison.png")

def plot_overall_judge_ranking(df):
    """Plots a horizontal bar chart for the overall average judge score."""
    overall_df = df[['Judge: Overall Avg']].copy().sort_values(by='Judge: Overall Avg', ascending=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = overall_df.plot(kind='barh', figsize=(12, 8), colormap='cividis')
    
    plt.title('Model Ranking by Overall Average Judge Score', fontsize=16, fontweight='bold')
    plt.xlabel('Average Score (1-5)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.legend().set_visible(False)
    plt.xlim(0, 5)
    
    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10, padding=5)
        
    plt.tight_layout()
    plt.savefig("overall_judge_ranking.png", dpi=300)
    print("Saved plot to overall_judge_ranking.png")


if __name__ == '__main__':
    # Paste the markdown table here
    markdown_table = """
| Model / Configuration                | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 | NLI Entailment (↑) | NLI Contradiction (↓) | Judge: Relevance (1-5) | Judge: Faithfulness (1-5) | Judge: Coherence (1-5) | Judge: Conciseness (1-5) | **Judge: Overall Avg** |
| :----------------------------------- | :------ | :------ | :------ | :----------- | :----------------- | :-------------------- | :--------------------- | :------------------------ | :--------------------- | :----------------------- | :--------------------- |
| **Supervised Extractive Models**     |         |         |         |              |                    |                       |                        |                           |                        |                          |                        |
| `BERTSum (Fine-tuned)`               | **0.7634**  | **0.7035**  | **0.4993**  | **0.9308**   | **0.9843**         | 0.0095                | 2.86                   | 4.90                      | 4.48                   | 4.24                     | 4.12                   |
| `SVO-BERTSum (Fine-tuned)`           | 0.7512  | 0.6883  | 0.4845  | 0.9282       | **0.9843**         | **0.0095**                | 2.80                   | **4.92**                      | 4.42                   | 4.22                     | 4.09                   |
| `SVO-XGBoost (Supervised)`           | 0.3938  | 0.3235  | 0.3126  | 0.8849       | **0.9846**         | **0.0037**                | 2.88                   | 4.84                      | 4.68                   | 4.12                     | 4.13                   |
| **Unsupervised Extractive Models**   |         |         |         |              |                    |                       |                        |                           |                        |                          |                        |
| `TextRank_Baseline`                  | 0.4191  | 0.3878  | 0.3429  | 0.9019       | **0.9845**         | **0.0053**                | 1.48                   | 4.90                      | 4.34                   | 4.74                     | 3.87                   |
| `Advanced_Extractive (Balanced)`     | 0.3597  | 0.2823  | 0.2899  | 0.8779       | 0.9782             | **0.0053**                | 1.58                   | **4.96**                      | 4.34                   | **4.82**                     | 3.93                   |
| `Original_Paper_Method`              | 0.3892  | 0.3102  | 0.3093  | 0.8806       | 0.9827             | **0.0053**                | 1.84                   | 4.92                      | 4.36                   | 4.72                     | 3.96                   |
| **Abstractive & Hybrid Models**      |         |         |         |              |                    |                       |                        |                           |                        |                          |                        |
| `Hybrid (Faithfulness-Guaranteed)`   | 0.3534  | 0.2567  | 0.2728  | 0.8793       | **0.9833**         | **0.0000**                | 2.26                   | 4.88                      | 4.60                   | 4.10                     | 3.96                   |
| `Hybrid (Original)`                  | 0.3172  | 0.1078  | 0.1785  | 0.8609       | 0.4410             | 0.0805                | **3.18**                   | 4.92                      | **4.90**                   | **4.90**                     | **4.48**                   |
| `LLM_Only (Unsafe)`                  | 0.3016  | 0.1027  | 0.1686  | 0.8610       | 0.1393             | 0.0703                | 1.00                   | 1.02                      | 2.76                   | 2.94                     | 1.93                   |
    """
    
    # Prepare data
    results_df = clean_and_prepare_data(markdown_table)
    
    # Generate plots
    plot_rouge_scores(results_df)
    plot_semantic_scores(results_df)
    plot_judge_scores(results_df)
    plot_overall_judge_ranking(results_df)

    print("\nAll visualizations have been generated.")