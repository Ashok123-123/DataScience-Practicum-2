"""
config.py — Central configuration for the Fake News Dataset Pipeline
=====================================================================
Edit TARGET_RECORDS, API keys, and flags here before running.
"""

import os

# ── Output paths ───────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
OUTPUT_CSV      = os.path.join(DATA_DIR, "fakenews_dataset.csv")
CHECKPOINT_CSV  = os.path.join(DATA_DIR, "checkpoint.json")
AUDIT_REPORT_JSON = os.path.join(DATA_DIR, "dataset_audit_report.json")

# ── Dataset targets ────────────────────────────────────────────────────────────
TARGET_RECORDS      = 150_000   # aim for 75K (between 50K–150K)
BALANCE_RATIO       = 0.5      # 50 % fake / 50 % real
MIN_TEXT_LENGTH     = 20       # chars — shorter rows are dropped
MAX_TEXT_LENGTH     = 5_000    # chars — very long rows are truncated

# ── API keys (set via env-vars or paste here) ──────────────────────────────────
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")   # optional

# ── HTTP settings ──────────────────────────────────────────────────────────────
REQUEST_TIMEOUT      = 15      # seconds
MAX_RETRIES          = 3
RETRY_BACKOFF        = 2.0     # exponential multiplier
RATE_LIMIT_DELAY     = 1.5     # seconds between requests per source
MAX_WORKERS          = 6       # parallel threads

# ── Label mapping (lowercase verdict → canonical label) ───────────────────────
FAKE_KEYWORDS = {
    "false", "fake", "pants on fire", "misleading", "misinformation",
    "disinformation", "debunked", "not true", "fabricated", "hoax",
    "scam", "manipulated", "incorrect", "inaccurate", "satire",
    "unverified", "disputed", "no evidence",
    "mostly false", "fiction", "rumor", "myth",
}

REAL_KEYWORDS = {
    "true", "correct", "accurate", "verified", "confirmed",
    "mostly true", "real", "legitimate", "credible",
    "proven", "valid",
}

UNCERTAIN_KEYWORDS = {
    "unproven", "unclear", "unknown", "mixed", "complicated",
    "needs context", "outdated", "exaggerated", "partly", "half true",
}

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL     = "INFO"
LOG_FILE      = os.path.join(LOG_DIR, "pipeline.log")
LOG_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FMT  = "%Y-%m-%d %H:%M:%S"
