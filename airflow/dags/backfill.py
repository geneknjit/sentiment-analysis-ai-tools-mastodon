#!/usr/bin/env python3
"""
backfill.py — One-off historical backfill for the Mastodon sentiment pipeline.

NOT a DAG. This is a standalone script that pages backward through hashtag
history on fosstodon.org, applies the same cleaning + VADER sentiment scoring
as the live DAG, and inserts rows tagged with collection_method='backfill'.

Run from inside the airflow-worker container:

    docker-compose exec airflow-worker python /opt/airflow/dags/backfill.py

Safe to interrupt (Ctrl+C) and re-run — ON CONFLICT DO NOTHING prevents dupes.
"""

import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from html import unescape

import psycopg2
from bs4 import BeautifulSoup
from mastodon import Mastodon
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# === Configuration ===
TARGET_DAYS_BACK = 90        # how far back to attempt
MAX_POSTS_PER_HASHTAG = 2000 # safety cap so a single tag can't run away
PAGE_SIZE = 40
SLEEP_BETWEEN_CALLS = 1.0    # seconds, well under the 300/5min rate limit
API_BASE_URL = 'https://fosstodon.org'

TOPICS = {
    'chatgpt': ['chatgpt', 'openai', 'gpt4'],
    'claude':  ['claude', 'anthropic'],
    'gemini':  ['gemini', 'bard'],
    'copilot': ['githubcopilot', 'copilot'],
}

DB_CONFIG = {
    'host':     'postgres',
    'port':     5432,
    'user':     'airflow',
    'password': 'airflow',
    'dbname':   'mastodon_sentiment_db',
}


# === Cleaning + scoring (mirrors the DAG) ===

def clean_text(raw_html):
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, 'html.parser')
    text = unescape(soup.get_text())
    text = ' '.join(text.split())
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    return text


def label_sentiment(compound):
    if compound >= 0.05:
        return 'positive'
    if compound <= -0.05:
        return 'negative'
    return 'neutral'


# === Schema migration (idempotent) ===

def ensure_schema(cur):
    """Add collection_method column if it doesn't exist; mark old rows 'live'."""
    cur.execute("""
        ALTER TABLE sentiment_data
        ADD COLUMN IF NOT EXISTS collection_method TEXT;
    """)
    cur.execute("""
        UPDATE sentiment_data
        SET collection_method = 'live'
        WHERE collection_method IS NULL;
    """)


# === Main backfill ===

def main():
    token = os.getenv('MASTODON_ACCESS_TOKEN')
    if not token:
        print("ERROR: MASTODON_ACCESS_TOKEN env var not set.", file=sys.stderr)
        print("       The script reads this from the worker's environment,", file=sys.stderr)
        print("       which is wired up via docker-compose.", file=sys.stderr)
        sys.exit(1)

    mastodon = Mastodon(access_token=token, api_base_url=API_BASE_URL)
    analyzer = SentimentIntensityAnalyzer()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    ensure_schema(cur)
    conn.commit()
    print("Schema ready (collection_method column ensured).")

    cutoff = datetime.now(timezone.utc) - timedelta(days=TARGET_DAYS_BACK)
    print(f"Backfilling posts created on or after {cutoff.isoformat()[:10]}")
    print(f"Topics: {list(TOPICS.keys())}\n")

    total_inserted = 0
    total_fetched = 0

    for topic, hashtags in TOPICS.items():
        for tag in hashtags:
            print(f"--- {topic}/#{tag} ---")
            max_id = None
            page = 0
            inserted_for_tag = 0
            fetched_for_tag = 0

            while inserted_for_tag < MAX_POSTS_PER_HASHTAG:
                page += 1
                try:
                    posts = mastodon.timeline_hashtag(tag, limit=PAGE_SIZE, max_id=max_id)
                except Exception as e:
                    print(f"  Page {page} failed: {e}")
                    break

                if not posts:
                    print(f"  No more posts available (page {page}).")
                    break

                fetched_for_tag += len(posts)
                total_fetched += len(posts)

                page_inserted = 0
                for post in posts:
                    if post['created_at'] < cutoff:
                        continue
                    if post.get('language') != 'en':
                        continue

                    cleaned = clean_text(post['content'])
                    if len(cleaned.split()) < 3:
                        continue

                    scores = analyzer.polarity_scores(cleaned)

                    cur.execute("""
                        INSERT INTO sentiment_data (
                            post_id, topic, hashtag, created_at,
                            content, clean_content,
                            favourites, reblogs, replies,
                            compound_score, pos_score, neg_score, neu_score,
                            sentiment_category, collection_method
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (post_id) DO NOTHING
                    """, (
                        str(post['id']),
                        topic,
                        tag,
                        post['created_at'],
                        post['content'],
                        cleaned,
                        post['favourites_count'],
                        post['reblogs_count'],
                        post['replies_count'],
                        scores['compound'],
                        scores['pos'],
                        scores['neg'],
                        scores['neu'],
                        label_sentiment(scores['compound']),
                        'backfill',
                    ))
                    if cur.rowcount > 0:
                        page_inserted += 1

                conn.commit()
                inserted_for_tag += page_inserted
                total_inserted += page_inserted

                oldest = min(p['created_at'] for p in posts)
                print(f"  Page {page}: fetched {len(posts):3d}, "
                      f"inserted {page_inserted:3d}, "
                      f"oldest {oldest.isoformat()[:10]}")

                if oldest < cutoff:
                    print(f"  Reached cutoff date.")
                    break

                # Page back: use the smallest id from this batch
                max_id = min(int(p['id']) for p in posts)
                time.sleep(SLEEP_BETWEEN_CALLS)

            print(f"  Done: inserted {inserted_for_tag} new rows for #{tag}\n")

    cur.close()
    conn.close()

    print("=" * 50)
    print("Backfill complete.")
    print(f"  Total posts fetched:  {total_fetched}")
    print(f"  Total rows inserted:  {total_inserted}")
    print(f"  (Rows skipped: not English, too short, before cutoff, or already in DB.)")


if __name__ == '__main__':
    main()
