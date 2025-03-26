from __future__ import annotations
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
from datasets import load_dataset

class FSUChemRxivQest(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FSUChemRxivQest",
        dataset={
            "path": "mahmoudamiri/FSUChemRxivQest",
            "revision": "main",
        },
        description="A retrieval dataset for FSUChemRxivQest.",
        reference="https://huggingface.co/datasets/mahmoudamiri/FSUChemRxivQest",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" 
        @dataset{FSUChemRxivQest,
        title={FSUChemRxivQest Dataset},
        author={Mahmoud Amiri},
        year={2024},
        publisher={Hugging Face},
        url={https://huggingface.co/datasets/mahmoudamiri/FSUChemRxivQest}
        } 
        """,
    )

    def load_data(self, **kwargs):
        """Load dataset from Hugging Face while keeping splits separate."""
        try:
            # Load each dataset split separately to avoid schema merging issues
            corpus = load_dataset(self.metadata.dataset["path"], name="corpus", split="corpus")
            queries = load_dataset(self.metadata.dataset["path"], name="queries", split="corpus")
            qrels = load_dataset(self.metadata.dataset["path"], name="qrels", split="corpus")


            # Debugging: Print actual column names
            print("📂 Corpus Columns:", corpus.column_names)
            print("📂 Queries Columns:", queries.column_names)
            print("📂 Qrels Columns:", qrels.column_names)

            # Ensure correct column names
            expected_corpus = {"id", "title", "text"}
            expected_queries = {"id", "text"}
            expected_qrels = {"query_id", "doc_id", "relevance"}

            if set(corpus.column_names) != expected_corpus:
                raise ValueError(f"❌ Corpus mismatch: Expected {expected_corpus}, got {corpus.column_names}")

            if set(queries.column_names) != expected_queries:
                raise ValueError(f"❌ Queries mismatch: Expected {expected_queries}, got {queries.column_names}")

            if set(qrels.column_names) != expected_qrels:
                raise ValueError(f"❌ Qrels mismatch: Expected {expected_qrels}, got {qrels.column_names}")

            self.dataset = {"corpus": corpus, "queries": queries, "qrels": qrels}
            print(f"✅ Successfully loaded dataset: {self.metadata.name}")

        except Exception as e:
            print(f"❌ Error loading dataset {self.metadata.name}: {e}")
            self.dataset = None
