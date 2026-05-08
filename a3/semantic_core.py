from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CORPUS = """
Language lets people describe the world, remember events, and share plans with
one another. In a small research library, students read articles about machines,
music, cities, weather, and history. The librarian notices that some words often
appear together. A scientist writes about experiments, data, models, computers,
and predictions. A traveler writes about trains, stations, maps, rivers, hotels,
and streets. A musician writes about rhythm, melody, instruments, voices, and
performance. Even when two sentences do not use exactly the same words, they may
carry related meanings. A sentence about a computer program that learns from
data is close to a sentence about an intelligent model that studies examples.

Traditional text representation begins by counting. If a document mentions
computer, algorithm, and model many times, those terms become important signals.
TF-IDF reduces the influence of common words and highlights words that are
specific to a document. Matrix factorization methods such as latent semantic
analysis compress the term document matrix into a smaller space. Words that
appear in similar contexts can move near each other in that space, even if they
are not identical. Neural methods continue this idea by training vectors from
context. Word2Vec learns useful word embeddings by predicting a word from
neighboring words or predicting neighboring words from a target word.

Embeddings make it possible to measure semantic similarity with cosine distance.
The words king and queen are close because they share royal contexts, while the
difference between man and woman can describe a gender direction in the vector
space. GloVe learns from global co-occurrence statistics and often supports
analogies such as king minus man plus woman. FastText improves robustness by
building word vectors from character n-grams. Because of this subword structure,
it can create a vector for a misspelled or unseen word such as computeer. Sentence
vectors can be built by averaging word vectors. This simple pooling method loses
word order, but it gives a quick global meaning representation for comparing two
longer sentences.
"""


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


@dataclass(frozen=True)
class TfidfResult:
    documents: list[str]
    matrix: pd.DataFrame
    top_keywords: pd.DataFrame


def tokenize_sentences(text: str) -> list[str]:
    """Sentence splitting with an NLTK path and a regex fallback."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_words(text: str) -> list[str]:
    return [token.lower().strip("'") for token in TOKEN_RE.findall(text or "")]


def normalized_documents(text: str) -> list[str]:
    docs = tokenize_sentences(text)
    return [doc for doc in docs if tokenize_words(doc)]


def tokenized_corpus(text: str) -> list[list[str]]:
    return [tokens for tokens in (tokenize_words(doc) for doc in normalized_documents(text)) if tokens]


def compute_tfidf(text: str, max_features: int = 80) -> TfidfResult:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = normalized_documents(text)
    if len(docs) < 2:
        raise ValueError("Please provide at least two English sentences for TF-IDF analysis.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=terms)
    tfidf_df.index = [f"Doc {idx + 1}" for idx in range(len(docs))]

    weights = np.asarray(matrix.mean(axis=0)).ravel()
    top_idx = np.argsort(weights)[::-1][:5]
    top_keywords = pd.DataFrame(
        {
            "keyword": [terms[idx] for idx in top_idx],
            "mean_tfidf": [round(float(weights[idx]), 4) for idx in top_idx],
        }
    )
    return TfidfResult(documents=docs, matrix=tfidf_df, top_keywords=top_keywords)


def compute_lsa_coordinates(
    text: str,
    matrix_type: str = "tfidf",
    max_features: int = 60,
) -> pd.DataFrame:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    docs = normalized_documents(text)
    if len(docs) < 2:
        raise ValueError("LSA needs at least two sentence documents.")

    vectorizer_cls = TfidfVectorizer if matrix_type == "tfidf" else CountVectorizer
    vectorizer = vectorizer_cls(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
        max_features=max_features,
    )
    doc_term = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    term_doc = doc_term.T

    if term_doc.shape[0] < 2 or term_doc.shape[1] < 1:
        raise ValueError("Not enough vocabulary for LSA visualization.")

    n_components = min(2, term_doc.shape[1])
    coords = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(term_doc)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])

    return pd.DataFrame({"word": terms, "x": coords[:, 0], "y": coords[:, 1]})


def train_word2vec(
    sentences: list[list[str]],
    sg: int,
    window: int,
    vector_size: int = 50,
    epochs: int = 120,
    seed: int = 42,
):
    if not sentences:
        raise ValueError("No valid tokens found for Word2Vec training.")

    try:
        from gensim.models import Word2Vec

        return Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=1,
            workers=1,
            sg=sg,
            epochs=epochs,
            seed=seed,
        )
    except Exception:
        return SimpleEmbeddingModel(sentences, window=window, vector_size=vector_size, seed=seed, mode="word2vec")


def train_fasttext(
    sentences: list[list[str]],
    window: int,
    vector_size: int = 50,
    epochs: int = 120,
    seed: int = 42,
):
    if not sentences:
        raise ValueError("No valid tokens found for FastText training.")

    try:
        from gensim.models import FastText

        return FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=1,
            workers=1,
            sg=1,
            min_n=3,
            max_n=5,
            epochs=epochs,
            seed=seed,
        )
    except Exception:
        return SimpleEmbeddingModel(sentences, window=window, vector_size=vector_size, seed=seed, mode="fasttext")


class SimpleKeyedVectors:
    """Tiny co-occurrence embedding fallback for cloud environments without gensim."""

    def __init__(self, sentences: list[list[str]], window: int, vector_size: int, seed: int, mode: str) -> None:
        self.mode = mode
        self.vector_size = vector_size
        vocab = sorted({token for sentence in sentences for token in sentence})
        self.key_to_index = {word: idx for idx, word in enumerate(vocab)}
        rng = np.random.default_rng(seed)
        random_basis = {word: rng.normal(0, 1, vector_size) for word in vocab}
        vectors: dict[str, np.ndarray] = {word: np.zeros(vector_size, dtype=float) for word in vocab}

        for sentence in sentences:
            for idx, word in enumerate(sentence):
                left = max(0, idx - window)
                right = min(len(sentence), idx + window + 1)
                for ctx in sentence[left:idx] + sentence[idx + 1 : right]:
                    vectors[word] += random_basis[ctx]

        for word, vector in vectors.items():
            norm = np.linalg.norm(vector)
            if norm == 0:
                vector = random_basis[word]
                norm = np.linalg.norm(vector)
            vectors[word] = vector / max(norm, 1e-12)

        self.vectors = vectors
        self.ngram_vectors = self._build_ngram_vectors(vectors) if mode == "fasttext" else {}

    def _build_ngram_vectors(self, vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for word, vector in vectors.items():
            for ngram in char_ngrams(word):
                buckets[ngram].append(vector)
        return {ngram: np.mean(values, axis=0) for ngram, values in buckets.items()}

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __getitem__(self, word: str) -> np.ndarray:
        word = word.lower().strip()
        if word in self.vectors:
            return self.vectors[word]
        if self.mode != "fasttext":
            raise KeyError(word)

        pieces = [self.ngram_vectors[ngram] for ngram in char_ngrams(word) if ngram in self.ngram_vectors]
        if pieces:
            vector = np.mean(pieces, axis=0)
            norm = np.linalg.norm(vector)
            return vector / max(norm, 1e-12)
        raise KeyError(word)

    def most_similar(self, positive: Iterable = (), negative: Iterable = (), topn: int = 5):
        if isinstance(positive, str):
            positive = [positive]
        else:
            positive = list(positive or [])
        if isinstance(negative, str):
            negative = [negative]
        else:
            negative = list(negative or [])
        query = np.zeros(self.vector_size, dtype=float)
        excluded = set()
        for item in positive:
            if isinstance(item, str):
                excluded.add(item)
                query += self[item]
            else:
                query += np.asarray(item, dtype=float)
        for item in negative:
            if isinstance(item, str):
                excluded.add(item)
                query -= self[item]
            else:
                query -= np.asarray(item, dtype=float)

        scored = []
        for word, vector in self.vectors.items():
            if word in excluded:
                continue
            score = cosine_similarity(query, vector)
            if score is not None:
                scored.append((word, score))
        return sorted(scored, key=lambda row: row[1], reverse=True)[:topn]

    def similarity(self, word_a: str, word_b: str) -> float:
        return float(cosine_similarity(self[word_a], self[word_b]))


class SimpleEmbeddingModel:
    def __init__(self, sentences: list[list[str]], window: int, vector_size: int, seed: int, mode: str) -> None:
        self.wv = SimpleKeyedVectors(sentences, window=window, vector_size=vector_size, seed=seed, mode=mode)


def char_ngrams(word: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    wrapped = f"<{word.lower()}>"
    grams = []
    for n in range(min_n, max_n + 1):
        grams.extend(wrapped[idx : idx + n] for idx in range(max(0, len(wrapped) - n + 1)))
    return grams


def top_similar(model, word: str, topn: int = 5) -> pd.DataFrame:
    word = (word or "").strip().lower()
    if not word:
        raise ValueError("Please input a word.")
    if word not in model.wv.key_to_index:
        raise KeyError(word)

    rows = model.wv.most_similar(word, topn=topn)
    return pd.DataFrame({"word": [row[0] for row in rows], "cosine_similarity": [round(float(row[1]), 4) for row in rows]})


def most_similar_from_vector(model, vector: np.ndarray, topn: int = 5) -> pd.DataFrame:
    rows = model.wv.most_similar(positive=[vector], topn=topn)
    return pd.DataFrame({"word": [row[0] for row in rows], "cosine_similarity": [round(float(row[1]), 4) for row in rows]})


def average_sentence_vector(sentence: str, keyed_vectors) -> np.ndarray | None:
    vectors = []
    for token in tokenize_words(sentence):
        try:
            vectors.append(np.asarray(keyed_vectors[token], dtype=float))
        except Exception:
            continue
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def cosine_similarity(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float | None:
    if vec_a is None or vec_b is None:
        return None
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return None
    return float(np.dot(vec_a, vec_b) / denom)


class TinyAnalogyVectors:
    """Small deterministic fallback used only when GloVe cannot be downloaded."""

    def __init__(self) -> None:
        raw = {
            "man": [1, 0, 0, 0, 0],
            "woman": [-1, 0, 0, 0, 0],
            "king": [1, 1, 0, 0, 0],
            "queen": [-1, 1, 0, 0, 0],
            "france": [0, 0, 1, 0, 0],
            "paris": [0, 0, 1, 1, 0],
            "china": [0, 0, 2, 0, 0],
            "beijing": [0, 0, 2, 1, 0],
            "japan": [0, 0, 3, 0, 0],
            "tokyo": [0, 0, 3, 1, 0],
            "computer": [0, 0, 0, 0, 1],
            "algorithm": [0, 0, 0, 0.2, 1],
            "music": [0, 0, 0, 2, 0],
            "melody": [0, 0, 0, 2.1, 0],
        }
        self.vectors = {key: np.asarray(value, dtype=float) for key, value in raw.items()}
        self.key_to_index = {key: idx for idx, key in enumerate(self.vectors)}

    def __contains__(self, key: str) -> bool:
        return key in self.vectors

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[key.lower()]

    def _resolve(self, item) -> np.ndarray:
        if isinstance(item, str):
            return self[item]
        return np.asarray(item, dtype=float)

    def most_similar(self, positive: Iterable = (), negative: Iterable = (), topn: int = 5):
        positive = list(positive or [])
        negative = list(negative or [])
        query = sum((self._resolve(item) for item in positive), np.zeros(5))
        query -= sum((self._resolve(item) for item in negative), np.zeros(5))
        excluded = {item.lower() for item in positive + negative if isinstance(item, str)}

        scored = []
        for word, vector in self.vectors.items():
            if word in excluded:
                continue
            score = cosine_similarity(query, vector)
            if score is not None and not math.isnan(score):
                scored.append((word, score))
        return sorted(scored, key=lambda row: row[1], reverse=True)[:topn]

    def similarity(self, word_a: str, word_b: str) -> float:
        return float(cosine_similarity(self[word_a.lower()], self[word_b.lower()]))


def analogy_table(keyed_vectors, word_a: str, word_b: str, word_c: str, topn: int = 5) -> pd.DataFrame:
    words = [word_a.lower().strip(), word_b.lower().strip(), word_c.lower().strip()]
    missing = [word for word in words if word not in keyed_vectors.key_to_index]
    if missing:
        raise KeyError(", ".join(missing))
    rows = keyed_vectors.most_similar(positive=[words[0], words[2]], negative=[words[1]], topn=topn)
    return pd.DataFrame({"result": [row[0] for row in rows], "score": [round(float(row[1]), 4) for row in rows]})


def word_pair_similarity(keyed_vectors, word_a: str, word_b: str) -> float:
    words = [word_a.lower().strip(), word_b.lower().strip()]
    missing = [word for word in words if word not in keyed_vectors.key_to_index]
    if missing:
        raise KeyError(", ".join(missing))
    return float(keyed_vectors.similarity(words[0], words[1]))
