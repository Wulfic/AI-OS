"""Catalog of known curated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class KnownDataset:
    """Metadata for a known dataset."""
    name: str
    url: str
    approx_size_gb: float
    notes: str = ""


def known_datasets(max_size_gb: float = 15.0) -> List[KnownDataset]:
    """A curated list of popular NLP datasets under a size threshold.

    The canonical list: https://github.com/niderhoff/nlp-datasets
    Here we provide a small, practical subset with approximate sizes.
    """
    all_ds: List[KnownDataset] = [
        KnownDataset("ag_news", "https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv", 0.1, "AG News classification"),
        KnownDataset("dbpedia", "https://github.com/le-scientifique/torchDatasets/tree/master/dbpedia_csv", 0.3, "DBPedia ontology classification"),
        KnownDataset("yelp_review_polarity", "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz", 3.6, "Yelp polarity reviews"),
        KnownDataset("amazon_review_polarity", "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz", 4.0, "Amazon polarity reviews"),
        KnownDataset("wikitext-103", "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/", 0.6, "Language modeling"),
        KnownDataset("snli", "https://nlp.stanford.edu/projects/snli/", 0.3, "Stanford NLI"),
        KnownDataset("multi_nli", "https://cims.nyu.edu/~sbowman/multinli/", 0.4, "MultiNLI"),
        KnownDataset("squad_v1", "https://rajpurkar.github.io/SQuAD-explorer/", 0.2, "QA dataset"),
        KnownDataset("imdb_reviews", "https://ai.stanford.edu/~amaas/data/sentiment/", 0.25, "IMDB reviews"),
    ]
    return [d for d in all_ds if d.approx_size_gb <= float(max_size_gb)]
