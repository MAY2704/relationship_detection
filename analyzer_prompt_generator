import pandas as pd
import os
import glob
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
INPUT_DIR = 'banking_data_files'
DOT_OUTPUT_FILE = 'data_model.dot'
PROMPT_OUTPUT_FILE = 'copilot_prompt.txt'
UNIQUENESS_THRESHOLD = 0.95 
INCLUSION_THRESHOLD = 0.99 
NAME_SIMILARITY_THRESHOLD = 0.60 


def load_dataframes_from_csv(directory):
    """
    Loads dataframes from CSV files, reading only the top 100 lines 
    to quickly sample the data and improve performance.
    """
    data_frames = {}
    file_paths = glob.glob(os.path.join(directory, '*.csv'))
    if not file_paths:
        raise FileNotFoundError(f"No CSV files found in the directory: {directory}. Run data_generator.py first.")
    for file_path in file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # CHANGE 1: Load only the first 100 rows for analysis
        data_frames[file_name] = pd.read_csv(file_path, low_memory=False, nrows=100)
        
    print(f"Loaded samples (top 100 lines) from {len(data_frames)} files for analysis.")
    return data_frames

def preprocess_col_name(name):
    """Splits camelCase/snake_case names into tokens for better NLP analysis."""
    tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
    tokens = [t.lower() for t in tokens if t.lower() not in ['fk', 'id', 'key', 'num', 'nr']]
    return ' '.join(tokens)

def calculate_semantic_score(fk_col, pk_col):
    """
    Combines FuzzyWuzzy and Cosine Similarity into a single, consolidated 
    semantic score (0.0 to 1.0).
    """
    fk_clean = preprocess_col_name(fk_col)
    pk_clean = preprocess_col_name(pk_col)
    fuzzy_score = fuzz.partial_ratio(fk_col.lower(), pk_col.lower()) / 100.0 
    
    corpus = [fk_clean, pk_clean]
    if len(fk_clean) == 0 or len(pk_clean) == 0:
        cosine_sim_score = 0.0
    else:
        vectorizer = CountVectorizer().fit_transform(corpus)
        vectors = vectorizer.toarray()
        sim_matrix = cosine_similarity(vectors)
        cosine_sim_score = sim_matrix[0][1]

    weights = {'fuzzy': 0.6, 'cosine': 0.4} 
    consolidated_score = (
        weights['fuzzy'] * fuzzy_score +
        weights['cosine'] * cosine_sim_score
    )
    
    return consolidated_score, fuzzy_score, cosine_sim_score, 0.0 

def discover_relationships(data_frames):
    """
    Analyzes all columns to find PK candidates and FK relationships, incorporating
    the semantic score.
    """
    
    print("\n--- Step 1: Profiling and Identifying Primary Key Candidates (Based on 100-Row Sample) ---")
    pk_candidates = {}
    
    for file_name, df in data_frames.items():
        pk_candidates[file_name] = []
        for col in df.columns:
            col_series = df[col].dropna()
            
            if not col_series.empty:
                uniqueness_ratio = col_series.nunique() / len(col_series)
            else:
                uniqueness_ratio = 0
            
            # PK candidate must be highly unique
            if uniqueness_ratio >= UNIQUENESS_THRESHOLD:
                pk_candidates[file_name].append({
                    'col_name': col,
                    'uniqueness': uniqueness_ratio,
                    'unique_set': set(col_series)
                })
                print(f"   ✅ PK Candidate: {file_name}.{col} (Unique: {uniqueness_ratio:.2f})")
    
    print("\n--- Step 2: Checking Inclusion Dependencies and Semantic Score ---")
    discovered_relationships = []

    for fk_file, fk_df in data_frames.items():
        for fk_col in fk_df.columns:
            
            fk_series = fk_df[fk_col].dropna()
            if fk_series.empty: continue
            fk_values = set(fk_series)

            for pk_file, candidates in pk_candidates.items():
                if fk_file == pk_file: continue

                for candidate in candidates:
                    pk_col = candidate['col_name']
                    pk_values = candidate['unique_set']
                    
                    # A. Semantic Score Check
                    consolidated_score, fuzzy_score, cosine_score, wordnet_score = calculate_semantic_score(fk_col, pk_col)
                    if consolidated_score < NAME_SIMILARITY_THRESHOLD: continue 

                    # B. Data Type Check
                    if str(fk_df[fk_col].dtype) != str(data_frames[pk_file][pk_col].dtype): continue

                    # C. Inclusion Dependency Check
                    mismatched_values = len(fk_values - pk_values)
                    coverage_ratio = 1 - (mismatched_values / len(fk_values))

                    # D. Final Heuristic Check
                    if coverage_ratio >= INCLUSION_THRESHOLD:
                        relationship = {
                            'FK_Table': fk_file,
                            'FK_Column': fk_col,
                            'PK_Table': pk_file,
                            'PK_Column': pk_col,
                            'Coverage': coverage_ratio,
                            'Semantic_Score': consolidated_score,
                            'Fuzzy_Score': fuzzy_score,
                            'Cosine_Score': cosine_score
                        }
                        discovered_relationships.append(relationship)
                        print(f"   ➡️ Found: {fk_file}.{fk_col} -> {pk_file}.{pk_col} (Cov: {coverage_ratio:.4f}, Sem: {consolidated_score:.2f})")

    return pd.DataFrame(discovered_relationships), pk_candidates

def generate_dot_metadata(relationships_df, pk_candidates_dict, data_frames_dict, output_file):
    """
    Generates a DOT language file in the specified Crow's Foot format.
    """
    all_tables = set(data_frames_dict.keys())
    
    dot_content = [
        'digraph BankingSchema {', 
        '   rankdir=LR;',
        '   // Global attributes for edges (Crow\'s Foot notation)',
        '   edge [arrowhead="crow", arrowtail="normal", dir="both", fontsize=10];'
    ] 
    
    # 1. Define Nodes (Tables)
    dot_content.append('\n   // Table nodes')
    for table_name in sorted(list(all_tables)):
        pk_info = next((c['col_name'] for c in pk_candidates_dict.get(table_name, [])), 'N/A')
        dot_content.append(f'   {table_name} [label="{table_name}\\nPK: {pk_info}"];')
        
    # 2. Define Edges (Relationships)
    dot_content.append('\n   // Relationships (FK -> PK)')
    sorted_relationships = relationships_df.sort_values(by=['FK_Table', 'PK_Table'])

    for index, row in sorted_relationships.iterrows():
        label = f"{row['FK_Column']} → {row['PK_Column']}"
        dot_content.append(f'   {row["FK_Table"]} -> {row["PK_Table"]} [label="{label}"];')

    dot_content.append('}')
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(dot_content))
    
    print(f"\n✅ Visualization metadata (Semantic Search Model) saved to: {output_file}")

def generate_copilot_prompt(metadata_df, dot_file_path, output_file):
    """Generates and saves the Copilot prompt file, including the Semantic Search DOT context."""
    
    # 1. Read DOT file content
    try:
        with open(dot_file_path, 'r') as f:
            semantic_dot_content = f.read()
    except FileNotFoundError:
        semantic_dot_content = "// ERROR: Preliminary DOT file not found."

    metadata_list = metadata_df.to_dict('records')
    
    # CHANGE 2: Updated language to 'detect lineage' and 'infer data flow'
    prompt_text = (
        "I have a dataset sample (top 100 rows per file) for a banking system. The following DOT code (DOT A) was generated "
        "by a preliminary 'Semantic Search' relationship analysis, which attempted to **detect lineage** "
        "and infer data flow purely based on statistical data metrics and column name similarity.\n\n"
        "**Your task (as a world-class Data Architect) is to critique this preliminary lineage model, "
        "correct any design flaws (e.g., non-unique primary keys like using names as PKs), and "
        "generate a new, semantically superior DOT file (DOT B) for the banking schema.**\n\n"
        "The new model MUST adhere to best practices (e.g., using ID columns for PKs) "
        "and should be named 'CopilotSchema'. Use the simple arrowhead for the 'one' side (PK) "
        "and 'crow' for the 'many' side (FK) in your final DOT output.\n\n"
    )

    prompt_text += "## A. Preliminary Semantic Search DOT Model (Critique and Correct This):\n"
    prompt_text += "```dot\n"
    prompt_text += semantic_dot_content
    prompt_text += "\n```\n\n"

    prompt_text += "## B. Underlying Column Metadata (For Context and PK Selection):\n"
    for item in metadata_list:
        prompt_text += f"- Table: {item['Table']}, Column: {item['Column']}, Type: {item['DataType']}\n"
    
    prompt_text += "\n\n**CRITICAL INSTRUCTION:** Based on the metadata and your banking knowledge, provide the refined, correct, and complete DOT code for the 'CopilotSchema' that reflects best practices."
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        print(f"✅ Copilot prompt saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving prompt file: {e}")
    
    return prompt_text


if __name__ == "__main__":
    print("--- Starting Final Relationship Analysis ---")

    try:
        # 1. Load data and Discover Model
        banking_data = load_dataframes_from_csv(INPUT_DIR)
        discovered_model, pk_candidates_dict = discover_relationships(banking_data)
        
        # 2. Extract Metadata for Prompt
        metadata_records = []
        for file_name, df in banking_data.items():
            for col_name, dtype in df.dtypes.items():
                if 'int' in str(dtype) or 'float' in str(dtype):
                    # Recalculate uniqueness check using the 100-row sample
                    is_key = df[col_name].nunique() / len(df) > 0.9 
                    type_simple = 'Numeric ID/Key' if is_key else 'Numeric Data'
                elif 'date' in str(dtype) or 'time' in str(dtype):
                    type_simple = 'Timestamp/Date'
                else:
                    type_simple = 'Text/String'
                metadata_records.append({'Table': file_name, 'Column': col_name, 'DataType': type_simple})
        metadata_df = pd.DataFrame(metadata_records)


        if not discovered_model.empty:
            # 3. Generate DOT metadata (Semantic Search Model)
            generate_dot_metadata(discovered_model, pk_candidates_dict, banking_data, DOT_OUTPUT_FILE)
            
            # 4. Generate Copilot Prompt (including Semantic Search DOT)
            generate_copilot_prompt(metadata_df, DOT_OUTPUT_FILE, PROMPT_OUTPUT_FILE)
            
            # 5. Print Final Report
            print("\n" + "="*80)
            print("FINAL DISCOVERED DATA MODEL REPORT (Semantic Search)")
            print("="*80)
            output_df = discovered_model[['FK_Table', 'FK_Column', 'PK_Table', 'PK_Column', 'Coverage', 'Semantic_Score']].sort_values(by=['FK_Table', 'PK_Table'])
            output_df['Relationship'] = output_df['FK_Table'] + '.' + output_df['FK_Column'] + ' → ' + output_df['PK_Table'] + '.' + output_df['PK_Column']
            output_df['Coverage'] = (output_df['Coverage'] * 100).round(2).astype(str) + '%'
            output_df['Semantic_Score'] = (output_df['Semantic_Score'] * 100).round(2).round(2).astype(str) + '%'
            print(output_df[['Relationship', 'Coverage', 'Semantic_Score']].to_string(index=False))

        else:
            print("No strong relationships were discovered.")

    except FileNotFoundError as e:
        print(f"\n🚨 ERROR: {e}")
        print("Please ensure the 'banking_data_files' directory exists and contains CSVs (run data_generator.py first).")

    print("\n--- Analysis Complete ---")

