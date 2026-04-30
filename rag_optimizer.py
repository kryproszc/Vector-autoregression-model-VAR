from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import optuna
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    DEVICE,
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    BASE_CONFIG,
)

from rag_core import run_single_excel


N_TRIALS = 5


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}min {sec}s"
    if minutes:
        return f"{minutes}min {sec}s"
    return f"{sec}s"


def score_config(row: Dict[str, Any]) -> float:
    return (
        0.50 * row["accuracy_at_1"]
        + 0.25 * row["accuracy_at_3"]
        + 0.15 * row["recall_at_k"]
        + 0.10 * row["answer_token_f1"]
    )


def suggest_config(trial: optuna.Trial) -> Dict[str, Any]:
    config = deepcopy(BASE_CONFIG)

    config["chunk_max_words"] = trial.suggest_categorical(
        "chunk_max_words",
        [80, 100, 120, 150, 180, 220],
    )

    config["chunk_overlap_words"] = trial.suggest_categorical(
        "chunk_overlap_words",
        [10, 20, 30, 40, 50],
    )

    config["dense_top_k"] = trial.suggest_categorical(
        "dense_top_k",
        [20, 40, 60, 80],
    )

    config["bm25_top_k"] = trial.suggest_categorical(
        "bm25_top_k",
        [20, 40, 60, 80],
    )

    config["rerank_top_k"] = trial.suggest_categorical(
        "rerank_top_k",
        [10, 20, 40, 60],
    )

    config["final_top_k"] = trial.suggest_categorical(
        "final_top_k",
        [3, 5, 7],
    )

    config["bm25_weight"] = trial.suggest_float(
        "bm25_weight",
        0.6,
        2.5,
    )

    config["rrf_k"] = trial.suggest_categorical(
        "rrf_k",
        [20, 30, 60, 100],
    )

    config["answer_max_chunks"] = trial.suggest_int(
        "answer_max_chunks",
        1,
        4,
    )

    config["answer_sentences_per_chunk"] = trial.suggest_int(
        "answer_sentences_per_chunk",
        1,
        2,
    )

    if config["chunk_overlap_words"] >= config["chunk_max_words"]:
        raise optuna.TrialPruned()

    return config


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excel_files = sorted(DATA_DIR.glob("*.xlsx"))

    if not excel_files:
        raise FileNotFoundError(
            f"Brak plików .xlsx w folderze: {DATA_DIR}"
        )

    embedding_model_name = EMBEDDING_MODELS[0]
    reranker_model_name = RERANKER_MODELS[0]

    print("=" * 100)
    print("RAG OPTIMIZER — OPTUNA")
    print(f"Liczba prób        : {N_TRIALS}")
    print(f"Liczba plików Excel: {len(excel_files)}")
    print(f"Embedding          : {embedding_model_name}")
    print(f"Reranker           : {reranker_model_name}")
    print(f"Folder danych      : {DATA_DIR}")
    print(f"Folder wyników     : {OUTPUT_DIR}")
    print("=" * 100)

    print("\nŁadowanie modeli...")
    retriever = SentenceTransformer(embedding_model_name, device=DEVICE)
    reranker = CrossEncoder(reranker_model_name, device=DEVICE)

    all_trials_rows: List[Dict[str, Any]] = []
    all_questions_rows: List[Dict[str, Any]] = []

    start_all = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial)

        print("\n" + "-" * 100)
        print(f"TRIAL {trial.number}")
        print(config)
        print("-" * 100)

        trial_start = time.perf_counter()

        file_metrics_rows = []
        trial_question_rows = []

        for file_no, excel_path in enumerate(excel_files, start=1):
            print(f"Testuję plik {file_no}/{len(excel_files)}: {excel_path.name}")

            try:
                metrics, question_rows = run_single_excel(
                    input_xlsx=excel_path,
                    config=config,
                    retriever=retriever,
                    reranker=reranker,
                )

                file_metrics_rows.append(metrics)

                for qr in question_rows:
                    qr = dict(qr)
                    qr["trial_number"] = trial.number
                    qr["embedding_model"] = embedding_model_name
                    qr["reranker_model"] = reranker_model_name
                    qr.update(config)
                    trial_question_rows.append(qr)

                partial_score = score_config(metrics)
                trial.report(partial_score, step=file_no)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise

            except Exception as e:
                print(f"BŁĄD dla pliku {excel_path.name}: {e}")

                file_metrics_rows.append({
                    "file": excel_path.name,
                    "questions": 0,
                    "chunks": 0,
                    "accuracy_at_1": 0.0,
                    "accuracy_at_3": 0.0,
                    "recall_at_k": 0.0,
                    "answer_match": 0.0,
                    "answer_token_f1": 0.0,
                    "avg_question_seconds": 0.0,
                    "total_seconds": 0.0,
                    "error": str(e),
                })

        df_file_metrics = pd.DataFrame(file_metrics_rows)

        summary = {
            "trial_number": trial.number,
            "embedding_model": embedding_model_name,
            "reranker_model": reranker_model_name,
            **config,
            "files": len(excel_files),
            "questions": int(df_file_metrics["questions"].sum()),
            "avg_chunks": float(df_file_metrics["chunks"].mean()),
            "accuracy_at_1": float(df_file_metrics["accuracy_at_1"].mean()),
            "accuracy_at_3": float(df_file_metrics["accuracy_at_3"].mean()),
            "recall_at_k": float(df_file_metrics["recall_at_k"].mean()),
            "answer_match": float(df_file_metrics["answer_match"].mean()),
            "answer_token_f1": float(df_file_metrics["answer_token_f1"].mean()),
            "avg_question_seconds": float(df_file_metrics["avg_question_seconds"].mean()),
            "trial_total_seconds": time.perf_counter() - trial_start,
        }

        summary["agent_score"] = score_config(summary)

        all_trials_rows.append(summary)
        all_questions_rows.extend(trial_question_rows)

        trial.set_user_attr("summary", summary)
        trial.set_user_attr("config", config)

        partial_path = OUTPUT_DIR / "optuna_partial_results.xlsx"
        pd.DataFrame(all_trials_rows).to_excel(partial_path, index=False)

        print(
            f"WYNIK TRIAL {trial.number}: "
            f"ACC@1={summary['accuracy_at_1']:.3f} | "
            f"ACC@3={summary['accuracy_at_3']:.3f} | "
            f"Recall={summary['recall_at_k']:.3f} | "
            f"F1={summary['answer_token_f1']:.3f} | "
            f"Score={summary['agent_score']:.3f} | "
            f"Czas triala={format_seconds(summary['trial_total_seconds'])}"
        )

        return summary["agent_score"]

    sampler = optuna.samplers.TPESampler(seed=42)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    for i in range(N_TRIALS):
        study.optimize(objective, n_trials=1)

        elapsed_total = time.perf_counter() - start_all
        completed = i + 1
        avg_trial_time = elapsed_total / completed
        remaining_trials = N_TRIALS - completed
        estimated_remaining = avg_trial_time * remaining_trials

        print("\n" + "=" * 100)
        print(f"POSTĘP OPTUNA: {completed}/{N_TRIALS}")
        print(f"Czas od startu       : {format_seconds(elapsed_total)}")
        print(f"Średnio na trial     : {format_seconds(avg_trial_time)}")
        print(f"Szacowany czas końca : {format_seconds(estimated_remaining)}")
        print("=" * 100)

    df_trials = pd.DataFrame(all_trials_rows).sort_values(
        "agent_score",
        ascending=False,
    )

    df_questions = pd.DataFrame(all_questions_rows)

    best_row = df_trials.iloc[0].to_dict()

    config_keys = [
        "chunk_max_words",
        "chunk_overlap_words",
        "dense_top_k",
        "bm25_top_k",
        "rerank_top_k",
        "final_top_k",
        "rrf_k",
        "bm25_weight",
        "embedding_batch_size",
        "reranker_batch_size",
        "answer_max_chunks",
        "answer_sentences_per_chunk",
    ]

    best_config = {
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "config": {
            key: best_row[key]
            for key in config_keys
        },
        "metrics": {
            "agent_score": best_row["agent_score"],
            "accuracy_at_1": best_row["accuracy_at_1"],
            "accuracy_at_3": best_row["accuracy_at_3"],
            "recall_at_k": best_row["recall_at_k"],
            "answer_match": best_row["answer_match"],
            "answer_token_f1": best_row["answer_token_f1"],
            "avg_question_seconds": best_row["avg_question_seconds"],
        },
    }

    output_xlsx = OUTPUT_DIR / "rag_optimizer_optuna_results.xlsx"
    best_json = OUTPUT_DIR / "best_config_optuna.json"

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df_trials.to_excel(writer, sheet_name="Ranking_prob", index=False)

        if not df_questions.empty:
            df_questions.to_excel(
                writer,
                sheet_name="Pytania_szczegoly",
                index=False,
            )

    save_json(best_json, best_config)

    print("\n" + "=" * 100)
    print("ZAKOŃCZONO OPTUNA")
    print(f"Czas całości : {format_seconds(time.perf_counter() - start_all)}")
    print(f"Raport Excel : {output_xlsx}")
    print(f"Best config  : {best_json}")
    print("=" * 100)

    print("\nNAJLEPSZA KONFIGURACJA:")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()