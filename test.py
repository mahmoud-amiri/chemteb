from datasets import load_dataset

# Load subsets
corpus_data = load_dataset("BASF-AI/ChemNQRetrieval", "corpus")
queries_data = load_dataset("BASF-AI/ChemNQRetrieval", "queries")
test_data = load_dataset("BASF-AI/ChemNQRetrieval")  # Loads the default (test) split

# Print available subsets
print("Available subsets:", {
    "corpus": corpus_data.keys(),
    "queries": queries_data.keys(),
    "test": test_data.keys(),
})

# Print column names for each subset
print("Corpus columns:", corpus_data["corpus"].column_names)
print("Queries columns:", queries_data["queries"].column_names)
print("Test columns:", test_data["test"].column_names)  # FIXED: Use "test" instead of "default"
