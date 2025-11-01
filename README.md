# relationship_detection

Automated Data Lineage discovery & LLM prompt generation

This Python script performs automated data lineage inference on a directory of CSV files. It applies a hybrid heuristic methodology—combining statistical data profiling with semantic analysis—to discover potential Primary Key (PK) and Foreign Key (FK) relationships.

The primary function is to generate two artifacts: a preliminary schema visualization and a structured, contextual prompt for an AI model to critique and refine the data model.

# Method

Relationship discovery relies on four key heuristics applied to a 100-row data sample from each CSV:

Statistical Profiling: Identifies PK candidates based on high data uniqueness ($\ge 95\%$).

Inclusion Dependency: Validates referential integrity by requiring near-perfect coverage ($\ge 99\%$) of FK values within the PK set.

Semantic Scoring: A combined score ($\ge 60\%$) assessing column name relatedness, weighting FuzzyWuzzy (60%) and Cosine Similarity (40%).

Data Type Coercion: Mandatory data type equivalence between linking columns.

# Setup and Usage

Prerequisites

Requires Python 3 and the following libraries: pip install pandas scikit-learn fuzzywuzzy


# Data Structure

Place all target CSV files within the directory specified by INPUT_DIR (default: banking_data_files).

# Execution

python data_model_analyzer.py


# Outputs

data_model.dot: Preliminary schema in Graphviz DOT format for visualization.

copilot_prompt.txt: Structured input for a Data Architect LLM, containing the DOT code and column metadata for final validation and refinement.

# Configuration

| Constant                   | Description                                                             | Default Value         |
|---------------------------|-------------------------------------------------------------------------|-----------------------|
| INPUT_DIR                 | Directory containing the CSV files.                                     | banking_data_files    |
| DOT_OUTPUT_FILE           | File path for the Graphviz visualization output.                        | data_model.dot        |
| PROMPT_OUTPUT_FILE        | File path for the LLM prompt text.                                      | copilot_prompt.txt    |
| UNIQUENESS_THRESHOLD      | Minimum uniqueness ratio for PK candidate selection.                    | 0.95                  |
| INCLUSION_THRESHOLD       | Minimum required coverage percentage for referential integrity.         | 0.99                  |
| NAME_SIMILARITY_THRESHOLD | Minimum combined semantic score for a valid relationship.               | 0.60                  |
