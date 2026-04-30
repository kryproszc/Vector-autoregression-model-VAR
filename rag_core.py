from __future__ import annotations

import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if os.cpu_count():
    torch.set_num_threads(max(1, min(8, os.cpu_count())))


STOPWORDS_PL = {
    "a", "aby", "albo", "ale", "ani", "bez", "bo", "by", "być", "był", "była",
    "było", "były", "co", "czy", "dla", "do", "gdy", "i", "ich", "jak",
    "jaki", "jakie", "jest", "jego", "jej", "lub", "ma", "na", "nad", "nie",
    "o", "od", "oraz", "po", "pod", "przez", "przy", "się", "są", "ta",
    "tak", "te", "ten", "to", "w", "we", "z", "za", "ze", "że"
}

TOKEN_RE = re.compile(
    r"[a-ząćęłńóśźż0-9]+(?:[%./-][a-ząćęłńóśźż0-9]+)*",
    re.IGNORECASE,
)


def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\u00ad", "")
    text = text.replace("￾", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    raw = [m.group(0).casefold() for m in TOKEN_RE.finditer(text)]
    filtered = [
        t for t in raw
        if t not in STOPWORDS_PL and (len(t) > 2 or any(ch.isdigit() for ch in t))
    ]
    return filtered or raw or [text.casefold()]


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def word_chunks(text: str, max_words: int, overlap_words: int) -> List[str]:
    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(0, end - overlap_words)

    return chunks


def read_input_excel(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(path)

    required_sheets = {"Teksty", "Pytania", "Oczekiwane"}
    missing = required_sheets - set(xls.sheet_names)

    if missing:
        raise ValueError(f"{path.name}: brakuje arkuszy: {missing}")

    texts_df = pd.read_excel(path, sheet_name="Teksty")
    questions_df = pd.read_excel(path, sheet_name="Pytania")
    expected_df = pd.read_excel(path, sheet_name="Oczekiwane")

    texts_df.columns = [str(c).strip().lower() for c in texts_df.columns]
    questions_df.columns = [str(c).strip().lower() for c in questions_df.columns]
    expected_df.columns = [str(c).strip().lower() for c in expected_df.columns]

    if not {"page", "text"}.issubset(texts_df.columns):
        raise ValueError("Arkusz Teksty musi mieć kolumny: page, text")

    if not {"question_id", "question"}.issubset(questions_df.columns):
        raise ValueError("Arkusz Pytania musi mieć kolumny: question_id, question")

    if not {"question_id", "expected_answer", "expected_pages"}.issubset(expected_df.columns):
        raise ValueError("Arkusz Oczekiwane musi mieć kolumny: question_id, expected_answer, expected_pages")

    texts_df["text"] = texts_df["text"].apply(normalize_text)
    questions_df["question"] = questions_df["question"].apply(normalize_text)
    expected_df["expected_answer"] = expected_df["expected_answer"].apply(normalize_text)

    texts_df = texts_df[texts_df["text"].str.len() > 0].copy()
    questions_df = questions_df[questions_df["question"].str.len() > 0].copy()

    return texts_df, questions_df, expected_df


def parse_expected_pages(value: Any) -> List[int]:
    if pd.isna(value):
        return []

    nums = re.findall(r"\d+", str(value))
    return sorted({int(n) for n in nums})


def build_chunks_from_pages(texts_df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []

    for _, row in texts_df.iterrows():
        page = int(row["page"])
        page_text = normalize_text(row["text"])

        raw_paragraphs = re.split(r"\n{2,}|(?<=\.)\s+(?=§\s*\d+)", page_text)
        paragraphs = [normalize_text(p) for p in raw_paragraphs if normalize_text(p)]

        if not paragraphs:
            paragraphs = [page_text]

        chunk_no = 1

        for para_no, paragraph in enumerate(paragraphs, start=1):
            subchunks = word_chunks(
                paragraph,
                max_words=config["chunk_max_words"],
                overlap_words=config["chunk_overlap_words"],
            )

            for sub_no, chunk_text in enumerate(subchunks, start=1):
                chunk_id = f"p{page:03d}_par{para_no:03d}_c{chunk_no:03d}"

                records.append({
                    "chunk_id": chunk_id,
                    "page": page,
                    "paragraph_no": para_no,
                    "subchunk_no": sub_no,
                    "text": chunk_text,
                    "search_text": f"Strona {page}. {chunk_text}",
                })

                chunk_no += 1

    if not records:
        raise ValueError("Nie zbudowano żadnych chunków.")

    return records


def encode_documents(model: SentenceTransformer, texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=config["embedding_batch_size"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype="float32")


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    emb = model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype="float32")


def build_search_engine(records: List[Dict[str, Any]], retriever: SentenceTransformer, config: Dict[str, Any]) -> Dict[str, Any]:
    search_texts = [r["search_text"] for r in records]
    tokenized = [tokenize(t) for t in search_texts]

    bm25 = BM25Okapi(tokenized)
    embeddings = encode_documents(retriever, search_texts, config)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "records": records,
        "search_texts": search_texts,
        "tokenized": tokenized,
        "bm25": bm25,
        "embeddings": embeddings,
        "index": index,
    }


def reciprocal_rank_fusion(
    dense_indices: List[int],
    bm25_indices: List[int],
    k: int,
    bm25_weight: float,
) -> Tuple[List[int], Dict[int, float]]:
    scores = {}

    for rank, idx in enumerate(dense_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    for rank, idx in enumerate(bm25_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + bm25_weight * (1.0 / (k + rank))

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked, scores


def search(
    query: str,
    engine: Dict[str, Any],
    retriever: SentenceTransformer,
    reranker: CrossEncoder,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:

    records = engine["records"]
    index = engine["index"]
    bm25 = engine["bm25"]
    n = len(records)

    dense_top_k = min(config["dense_top_k"], n)
    bm25_top_k = min(config["bm25_top_k"], n)
    rerank_top_k = min(config["rerank_top_k"], n)

    q_emb = encode_query(retriever, query)
    dense_scores, dense_indices = index.search(q_emb, dense_top_k)

    dense_indices_list = [int(i) for i in dense_indices[0].tolist() if int(i) != -1]
    dense_scores_list = [float(s) for s in dense_scores[0].tolist()]

    dense_rank = {idx: rank for rank, idx in enumerate(dense_indices_list, start=1)}
    dense_score_map = {
        idx: score for idx, score in zip(dense_indices_list, dense_scores_list)
    }

    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = [
        int(i) for i in np.argsort(bm25_scores)[::-1][:bm25_top_k].tolist()
    ]

    bm25_rank = {idx: rank for rank, idx in enumerate(bm25_indices, start=1)}
    bm25_score_map = {idx: float(bm25_scores[idx]) for idx in bm25_indices}

    candidate_indices, rrf_scores = reciprocal_rank_fusion(
        dense_indices=dense_indices_list,
        bm25_indices=bm25_indices,
        k=config["rrf_k"],
        bm25_weight=config["bm25_weight"],
    )

    candidate_indices = candidate_indices[:rerank_top_k]

    if not candidate_indices:
        return []

    pairs = [(query, records[i]["search_text"]) for i in candidate_indices]

    rerank_scores = reranker.predict(
        pairs,
        batch_size=config["reranker_batch_size"],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    rerank_scores = np.asarray(rerank_scores).reshape(-1)

    results = []

    for idx, score in zip(candidate_indices, rerank_scores):
        rec = dict(records[idx])
        rec.update({
            "dense_rank": dense_rank.get(idx),
            "dense_score": dense_score_map.get(idx),
            "bm25_rank": bm25_rank.get(idx),
            "bm25_score": bm25_score_map.get(idx),
            "rrf_score": float(rrf_scores.get(idx, 0.0)),
            "rerank_score": float(score),
        })
        results.append(rec)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results


def build_predicted_answer(query: str, hits: List[Dict[str, Any]], reranker: CrossEncoder, config: Dict[str, Any]) -> str:
    selected_sentences = []

    for hit in hits[:config["answer_max_chunks"]]:
        sentences = split_sentences(hit["text"])
        pairs = [(query, s) for s in sentences if len(s) > 20]

        if not pairs:
            continue

        scores = reranker.predict(
            pairs,
            batch_size=config["reranker_batch_size"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        scored = list(zip(scores, [s for _, s in pairs]))
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, sentence in scored[:config["answer_sentences_per_chunk"]]:
            selected_sentences.append((float(score) + hit["rerank_score"], sentence))

    selected_sentences.sort(key=lambda x: x[0], reverse=True)

    final_sentences = []
    seen_tokens = []

    for _, sentence in selected_sentences:
        sent_norm = normalize_for_eval(sentence)
        sent_tokens = set(sent_norm.split())

        duplicate = False

        for prev_tokens in seen_tokens:
            if not sent_tokens or not prev_tokens:
                continue

            overlap = len(sent_tokens & prev_tokens) / max(1, min(len(sent_tokens), len(prev_tokens)))

            if overlap >= 0.75:
                duplicate = True
                break

        if not duplicate:
            final_sentences.append(sentence)
            seen_tokens.append(sent_tokens)

        if len(final_sentences) >= 2:
            break

    return " ".join(final_sentences).strip()


def normalize_for_eval(text: str) -> str:
    text = normalize_text(text).casefold()
    text = re.sub(r"[^\wąćęłńóśźż%]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_for_eval(pred).split()
    gold_tokens = normalize_for_eval(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    overlap = sum((pred_counter & gold_counter).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def expected_answer_match(pred: str, gold: str) -> bool:
    pred_n = normalize_for_eval(pred)
    gold_n = normalize_for_eval(gold)

    if not pred_n or not gold_n:
        return False

    if gold_n in pred_n:
        return True

    return token_f1(pred, gold) >= 0.55


def evaluate_hits(
    hits: List[Dict[str, Any]],
    expected_pages: List[int],
    expected_answer: str,
    predicted_answer: str,
    final_top_k: int,
) -> Dict[str, Any]:

    top_hits = hits[:final_top_k]
    hit_pages = [int(h["page"]) for h in top_hits]
    expected_pages_set = set(expected_pages)

    if expected_pages_set:
        acc1 = bool(hit_pages[:1] and hit_pages[0] in expected_pages_set)
        acc3 = any(p in expected_pages_set for p in hit_pages[:3])
        recall_k = len(expected_pages_set & set(hit_pages)) / len(expected_pages_set)

        first_k = None
        for i, p in enumerate(hit_pages, start=1):
            if p in expected_pages_set:
                first_k = i
                break
    else:
        acc1 = False
        acc3 = False
        recall_k = 0.0
        first_k = None

    answer_ok = expected_answer_match(predicted_answer, expected_answer)
    f1 = token_f1(predicted_answer, expected_answer)

    return {
        "accuracy_at_1": acc1,
        "accuracy_at_3": acc3,
        "recall_at_k": recall_k,
        "first_correct_rank": first_k,
        "answer_match": answer_ok,
        "answer_token_f1": f1,
    }


def run_single_excel(
    input_xlsx: Path,
    config: Dict[str, Any],
    retriever: SentenceTransformer,
    reranker: CrossEncoder,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    start = time.perf_counter()

    texts_df, questions_df, expected_df = read_input_excel(input_xlsx)

    expected_map = {
        str(row["question_id"]): {
            "expected_answer": normalize_text(row["expected_answer"]),
            "expected_pages": parse_expected_pages(row["expected_pages"]),
        }
        for _, row in expected_df.iterrows()
    }

    records = build_chunks_from_pages(texts_df, config)
    engine = build_search_engine(records, retriever, config)

    question_rows = []

    for row in questions_df.itertuples(index=False):
        question_id = str(getattr(row, "question_id"))
        question = normalize_text(getattr(row, "question"))

        expected = expected_map.get(
            question_id,
            {"expected_answer": "", "expected_pages": []},
        )

        expected_answer = expected["expected_answer"]
        expected_pages = expected["expected_pages"]

        q_start = time.perf_counter()
        hits = search(question, engine, retriever, reranker, config)
        q_elapsed = time.perf_counter() - q_start

        final_hits = hits[:config["final_top_k"]]
        predicted_answer = build_predicted_answer(question, final_hits, reranker, config)

        eval_result = evaluate_hits(
            hits=hits,
            expected_pages=expected_pages,
            expected_answer=expected_answer,
            predicted_answer=predicted_answer,
            final_top_k=config["final_top_k"],
        )

        top_pages = ";".join(str(int(h["page"])) for h in final_hits)

        question_rows.append({
            "file": input_xlsx.name,
            "question_id": question_id,
            "question": question,
            "expected_pages": ";".join(map(str, expected_pages)),
            "top_pages": top_pages,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "accuracy_at_1": eval_result["accuracy_at_1"],
            "accuracy_at_3": eval_result["accuracy_at_3"],
            "recall_at_k": eval_result["recall_at_k"],
            "first_correct_rank": eval_result["first_correct_rank"],
            "answer_match": eval_result["answer_match"],
            "answer_token_f1": eval_result["answer_token_f1"],
            "elapsed_seconds": q_elapsed,
        })

    df = pd.DataFrame(question_rows)

    metrics = {
        "file": input_xlsx.name,
        "questions": len(df),
        "chunks": len(records),
        "accuracy_at_1": float(df["accuracy_at_1"].mean()) if len(df) else 0.0,
        "accuracy_at_3": float(df["accuracy_at_3"].mean()) if len(df) else 0.0,
        "recall_at_k": float(df["recall_at_k"].mean()) if len(df) else 0.0,
        "answer_match": float(df["answer_match"].mean()) if len(df) else 0.0,
        "answer_token_f1": float(df["answer_token_f1"].mean()) if len(df) else 0.0,
        "avg_question_seconds": float(df["elapsed_seconds"].mean()) if len(df) else 0.0,
        "total_seconds": time.perf_counter() - start,
    }

    return metrics, question_rows