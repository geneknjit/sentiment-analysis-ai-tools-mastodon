#!/usr/bin/env python3
"""
ml_score.py — One-off transformer scoring for the Mastodon sentiment pipeline.

NOT a DAG. Adds ml_* columns to sentiment_data, then uses
cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa-base, 3-class) to
score every row that doesn't already have ML scores. Writes per-class
probabilities, a derived compound (positive - negative), and the argmax label.

Run from inside the airflow-worker container:

    docker-compose exec airflow-worker python /opt/airflow/dags/ml_score.py

Safe to interrupt and re-run — only scores rows where ml_sentiment_category
IS NULL, so progress isn't lost.

First run downloads ~500MB of model weights from huggingface.co. This is
cached in /home/airflow/.cache/huggingface/ inside the container; the cache
is lost when the container is rebuilt and re-downloaded on next first run.
"""

import sys
import time

import psycopg2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Local path inside the container, populated by host-side download_model.py.
# Loading from a local directory bypasses HuggingFace HTTP fetching, which
# is broken in transformers 4.18 against the current huggingface.co API.
MODEL_NAME = "/opt/airflow/data/hf_model"
LABELS = ['negative', 'neutral', 'positive']
BATCH_SIZE = 16          # small enough to fit in worker memory comfortably
MAX_LENGTH = 256         # truncation length for tokenizer
NUM_THREADS = 4          # be a good neighbor; leave cores free for airflow

DB_CONFIG = {
    'host':     'postgres',
    'port':     5432,
    'user':     'airflow',
    'password': 'airflow',
    'dbname':   'mastodon_sentiment_db',
}


def ensure_schema(cur):
    """Add ml_* columns if they don't exist. Idempotent."""
    cur.execute("""
        ALTER TABLE sentiment_data
        ADD COLUMN IF NOT EXISTS ml_negative REAL,
        ADD COLUMN IF NOT EXISTS ml_neutral REAL,
        ADD COLUMN IF NOT EXISTS ml_positive REAL,
        ADD COLUMN IF NOT EXISTS ml_compound REAL,
        ADD COLUMN IF NOT EXISTS ml_sentiment_category TEXT;
    """)


def main():
    torch.set_num_threads(NUM_THREADS)

    print(f"Loading model from: {MODEL_NAME}")
    print("(Files were pre-downloaded by host-side download_model.py.)")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s.\n")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    ensure_schema(cur)
    conn.commit()
    print("Schema ready (ml_* columns ensured).\n")

    cur.execute("""
        SELECT post_id, clean_content
        FROM sentiment_data
        WHERE ml_sentiment_category IS NULL
        ORDER BY created_at;
    """)
    rows = cur.fetchall()
    total = len(rows)

    if total == 0:
        print("All rows already scored. Nothing to do.")
        cur.close()
        conn.close()
        return

    print(f"Scoring {total} rows in batches of {BATCH_SIZE}...\n")

    start = time.time()
    scored = 0

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        texts = [(r[1] or "")[:2000] for r in batch]  # safety cap on text length
        ids = [r[0] for r in batch]

        with torch.no_grad():
            inputs = tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            )
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        for j, post_id in enumerate(ids):
            p = probs[j].tolist()
            neg, neu, pos = p[0], p[1], p[2]
            compound = pos - neg
            category = LABELS[max(range(3), key=lambda k: p[k])]

            cur.execute("""
                UPDATE sentiment_data
                SET ml_negative = %s,
                    ml_neutral = %s,
                    ml_positive = %s,
                    ml_compound = %s,
                    ml_sentiment_category = %s
                WHERE post_id = %s
            """, (neg, neu, pos, compound, category, post_id))

        conn.commit()
        scored += len(batch)
        elapsed = time.time() - start
        rate = scored / elapsed if elapsed > 0 else 0
        eta = (total - scored) / rate if rate > 0 else 0
        print(f"  Scored {scored}/{total} ({100 * scored / total:5.1f}%) "
              f"— {rate:5.1f} posts/sec, ETA {eta:5.0f}s")

    cur.close()
    conn.close()

    elapsed = time.time() - start
    print()
    print("=" * 50)
    print(f"Done. Scored {scored} rows in {elapsed:.1f}s "
          f"({scored / elapsed:.1f} posts/sec).")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved — re-run to continue.")
        sys.exit(1)
