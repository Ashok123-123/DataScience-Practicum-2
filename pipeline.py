#!/usr/bin/env python3
"""
pipeline.py — Master orchestrator for the Fake News Dataset Pipeline
=====================================================================

USAGE
-----
    # Full run (checks network, then runs live scraping)
  python pipeline.py

  # Force live scraping only (requires internet)
  python pipeline.py --mode live

  # Append to existing dataset (incremental run)
  python pipeline.py --append

FLAGS
-----
    --mode        : live | auto (default: auto)
    --target      : total records to collect (default: 75000)
  --append      : append to existing CSV instead of overwriting
  --no-balance  : skip class balancing step
  --keep-unknown: keep records labelled 'unknown'
"""

import argparse
import json
import os
import sys

# ensure local imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import (
    OUTPUT_CSV, CHECKPOINT_CSV, TARGET_RECORDS,
    DATA_DIR, LOG_DIR, AUDIT_REPORT_JSON,
)
from modules.logger   import get_logger
from modules.cleaner  import clean_text, label_record
from modules.balancer import balance_dataset, print_statistics
from modules.dataset_audit import audit_dataset, save_report
from modules.progress import StageTracker, ProgressBar


# ── Helpers ────────────────────────────────────────────────────────────────────

log = get_logger("pipeline")


def _check_network() -> bool:
    """Quick connectivity check — returns True if internet is reachable."""
    import socket
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            ("8.8.8.8", 53)
        )
        return True
    except Exception:
        return False


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_CSV):
        with open(CHECKPOINT_CSV) as f:
            return json.load(f)
    return {"seen_hashes": [], "runs": 0, "total_collected": 0}


def _save_checkpoint(data: dict):
    with open(CHECKPOINT_CSV, "w") as f:
        json.dump(data, f)


# ── Stage 1: Collect ───────────────────────────────────────────────────────────

def collect_data(mode: str, target: int) -> list[dict]:
    """
    Collect raw records from live scrapers.
    Returns list of raw dicts: {text, verdict, source}.
    """
    if mode == "auto" and not _check_network():
        log.error("No network available — live scraping requires internet connectivity.")
        sys.exit(1)

    log.info("Network available — running live scrapers")
    from modules.scrapers import run_all_scrapers
    records = run_all_scrapers(target_records=target)

    return records


# ── Stage 2: Clean & Label ─────────────────────────────────────────────────────

def process_records(raw: list[dict],
                    keep_unknown: bool = False) -> pd.DataFrame:
    """
    Apply cleaning pipeline to raw records.
    Returns a tidy DataFrame: text, label, confidence, source.
    """
    bar = ProgressBar(len(raw), prefix="Cleaning", suffix="records")
    bar.start()

    cleaned = []
    for i, rec in enumerate(raw):
        text    = clean_text(rec.get("text", "") or rec.get("headline", ""))
        verdict = rec.get("verdict", "")
        source  = rec.get("source", "unknown")

        # If label is already present and valid, keep it.
        if "label" in rec and rec["label"] in ("fake", "real"):
            label      = rec["label"]
            confidence = rec.get("confidence", 0.9)
        else:
            label, confidence = label_record(verdict)

        cleaned.append({
            "text":       text,
            "label":      label,
            "confidence": confidence,
            "source":     source,
        })
        if (i + 1) % 1000 == 0:
            bar.update(i + 1)

    bar.finish()

    df = pd.DataFrame(cleaned, columns=["text", "label", "confidence", "source"])

    # quality gates
    df = df[df["text"].str.len() >= 20]
    if not keep_unknown:
        df = df[df["label"] != "unknown"]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df


# ── Stage 3: Deduplicate (fingerprint) ────────────────────────────────────────

def dedup_dataframe(df: pd.DataFrame, seen_hashes: set) -> tuple[pd.DataFrame, set]:
    """Hash-based deduplication across runs."""
    from modules.cleaner import fingerprint
    mask = []
    for text in df["text"]:
        fp = fingerprint(text)
        if fp not in seen_hashes:
            seen_hashes.add(fp)
            mask.append(True)
        else:
            mask.append(False)
    return df[mask].reset_index(drop=True), seen_hashes


# ── Stage 4: Balance ───────────────────────────────────────────────────────────

def balance(df: pd.DataFrame, target: int) -> pd.DataFrame:
    return balance_dataset(df, target=target)


# ── Stage 5: Save ──────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, path: str, append_mode: bool):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if append_mode and os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["text"]
        )
        log.info(f"Appended — total rows now {len(df):,}")

    df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"Dataset saved → {path}  ({len(df):,} rows)")
    return df


def audit_and_report(df: pd.DataFrame, path: str = AUDIT_REPORT_JSON) -> dict:
    """Audit the dataset before save so leakage risks are visible."""
    report = audit_dataset(df)
    save_report(report, path)

    repeated_rows = report.get("repeated_template_rows", 0)
    synthetic_rows = report.get("rows_with_synthetic_patterns", 0)
    log.info(f"Dataset audit saved → {path}")
    log.info(
        "Audit summary | rows=%s | normalized_duplicates=%s | repeated_template_rows=%s | synthetic_pattern_rows=%s",
        f"{report.get('rows', 0):,}",
        f"{report.get('normalized_duplicate_texts', 0):,}",
        f"{repeated_rows:,}",
        f"{synthetic_rows:,}",
    )
    if repeated_rows or synthetic_rows:
        log.warning(
            "Potential leakage detected before save. Review %s before training models.",
            path,
        )
    return report


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(mode: str = "auto", target: int = TARGET_RECORDS,
                 append: bool = False,
                 no_balance: bool = False, keep_unknown: bool = False):

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    tracker = StageTracker()
    checkpoint = _load_checkpoint()
    seen_hashes = set(checkpoint.get("seen_hashes", []))

    log.info("=" * 60)
    log.info("Fake News Dataset Pipeline — START")
    log.info(f"  mode={mode}  target={target:,}  append={append}  no_balance={no_balance}")
    log.info("=" * 60)

    # ── 1. Collect ─────────────────────────────────────────────────────────────
    tracker.begin("Data Collection")
    raw = collect_data(mode=mode, target=target)
    tracker.end("Data Collection", records=len(raw))

    # ── 2. Clean & Label ───────────────────────────────────────────────────────
    tracker.begin("Cleaning & Labeling")
    df = process_records(raw, keep_unknown=keep_unknown)
    tracker.end("Cleaning & Labeling", rows=len(df))
    if df.empty:
        raise RuntimeError(
            "No labeled fake/real rows remained after cleaning. "
            "Set GOOGLE_FACTCHECK_API_KEY, or use scrapers with explicit ratings "
            "such as PolitiFact/Snopes. The collected rows were unlabeled/unknown."
        )

    # ── 3. Dedup ───────────────────────────────────────────────────────────────
    tracker.begin("Deduplication")
    before = len(df)
    df, seen_hashes = dedup_dataframe(df, seen_hashes)
    tracker.end("Deduplication", removed=before - len(df), remaining=len(df))

    # ── 4. Balance ─────────────────────────────────────────────────────────────
    if not no_balance:
        tracker.begin("Class Balancing")
        df = balance(df, target=target)
        tracker.end("Class Balancing", rows=len(df))

    # ── 5. Audit ───────────────────────────────────────────────────────────────
    tracker.begin("Dataset Audit")
    audit_report = audit_and_report(df)
    tracker.end(
        "Dataset Audit",
        repeated_templates=audit_report.get("repeated_template_rows", 0),
        synthetic_rows=audit_report.get("rows_with_synthetic_patterns", 0),
    )

    # ── 6. Save ────────────────────────────────────────────────────────────────
    tracker.begin("Save to CSV")
    df = save(df, OUTPUT_CSV, append_mode=append)
    tracker.end("Save to CSV", rows=len(df))

    # ── 7. Statistics ──────────────────────────────────────────────────────────
    stats = print_statistics(df, title="Final Dataset Statistics")

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    checkpoint["seen_hashes"]     = list(seen_hashes)[:500_000]  # cap size
    checkpoint["runs"]            += 1
    checkpoint["total_collected"] += len(df)
    _save_checkpoint(checkpoint)

    tracker.summary()
    log.info("Pipeline finished successfully.")
    return df, stats


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fake News Dataset Collection Pipeline"
    )
    parser.add_argument("--mode",          default="auto",
                        choices=["auto", "live"])
    parser.add_argument("--target",        type=int, default=TARGET_RECORDS)
    parser.add_argument("--append",        action="store_true")
    parser.add_argument("--no-balance",    action="store_true")
    parser.add_argument("--keep-unknown",  action="store_true")

    args = parser.parse_args()

    run_pipeline(
        mode         = args.mode,
        target       = args.target,
        append       = args.append,
        no_balance   = args.no_balance,
        keep_unknown = args.keep_unknown,
    )
