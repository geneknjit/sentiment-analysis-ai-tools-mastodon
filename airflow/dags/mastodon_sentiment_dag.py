from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pandas as pd
import os
import re

# Default arguments
default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 2),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'mastodon_sentiment_pipeline',
    default_args=default_args,
    description='Automated sentiment analysis with Postgres storage',
    schedule_interval=timedelta(minutes=15),
    catchup=False,
)


# Initialize database
def create_table():
    hook = PostgresHook(postgres_conn_id='postgre_sql')
    conn = hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_data (
            post_id TEXT PRIMARY KEY,
            topic TEXT,
            hashtag TEXT,
            created_at TIMESTAMP,
            content TEXT,
            clean_content TEXT,
            favourites INTEGER,
            reblogs INTEGER,
            replies INTEGER,
            compound_score REAL,
            pos_score REAL,
            neg_score REAL,
            neu_score REAL,
            sentiment_category TEXT,
            collection_method TEXT DEFAULT 'live',
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


# Task 1: Collect data
def collect_data(**context):
    """Fetch posts from Mastodon"""
    # Get token from environment (set in docker-compose.yaml)
    from mastodon import Mastodon
    mastodon = Mastodon(
        access_token=os.getenv('MASTODON_ACCESS_TOKEN'),
        api_base_url='https://fosstodon.org'
    )

    topics = {
        'chatgpt': ['chatgpt', 'openai', 'gpt4'],
        'claude': ['claude', 'anthropic'],
        'gemini': ['gemini', 'bard'],
        'copilot': ['githubcopilot', 'copilot'],
    }

    all_posts = []
    for topic, hashtags in topics.items():
        for tag in hashtags:
            try:
                posts = mastodon.timeline_hashtag(tag, limit=40)
                for post in posts:
                    if post.get('language') == 'en':
                        all_posts.append({
                            'post_id': post['id'],
                            'topic': topic,
                            'hashtag': tag,
                            'created_at': post['created_at'],
                            'content': post['content'],
                            'favourites': post['favourites_count'],
                            'reblogs': post['reblogs_count'],
                            'replies': post['replies_count']
                        })
                import time
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching {tag}: {e}")

    df = pd.DataFrame(all_posts)
    temp_path = '/tmp/raw_data_temp.csv'
    df.to_csv(temp_path, index=False)

    print(f"✅ Collected {len(df)} posts")
    return len(df)


# Task 2: Clean data
def clean_data(**context):
    """Clean HTML, URLs, mentions"""
    from bs4 import BeautifulSoup
    from html import unescape
    df = pd.read_csv('/tmp/raw_data_temp.csv')

    def clean_html(raw_html):
        if pd.isna(raw_html):
            return ""
        soup = BeautifulSoup(raw_html, 'html.parser')
        text = soup.get_text()
        text = unescape(text)
        text = ' '.join(text.split())
        return text

    def remove_urls(text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_mentions(text):
        return re.sub(r'@\w+', '', text)

    df['clean_content'] = df['content'].apply(clean_html)
    df['clean_content'] = df['clean_content'].apply(remove_urls)
    df['clean_content'] = df['clean_content'].apply(remove_mentions)
    df = df[df['clean_content'].str.split().str.len() >= 3]

    temp_path = '/tmp/cleaned_data_temp.csv'
    df.to_csv(temp_path, index=False)

    print(f"✅ Cleaned {len(df)} posts")
    return len(df)


# Task 3: Sentiment analysis
def analyze_sentiment(**context):
    """Apply VADER sentiment analysis"""
    df = pd.read_csv('/tmp/cleaned_data_temp.csv')
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    df['sentiment_scores'] = df['clean_content'].apply(lambda x: analyzer.polarity_scores(x))
    df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    df['pos_score'] = df['sentiment_scores'].apply(lambda x: x['pos'])
    df['neg_score'] = df['sentiment_scores'].apply(lambda x: x['neg'])
    df['neu_score'] = df['sentiment_scores'].apply(lambda x: x['neu'])
    df = df.drop('sentiment_scores', axis=1)

    def label_sentiment(compound):
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment_category'] = df['compound_score'].apply(label_sentiment)

    temp_path = '/tmp/scored_data_temp.csv'
    df.to_csv(temp_path, index=False)

    print(f"✅ Analyzed sentiment for {len(df)} posts")
    return len(df)


# Task 4: Store in Postgres
def store_in_database(**context):
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    df = pd.read_csv('/tmp/scored_data_temp.csv')

    hook = PostgresHook(postgres_conn_id='postgre_sql')
    conn = hook.get_conn()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO sentiment_data (
                post_id, topic, hashtag, created_at,
                content, clean_content,
                favourites, reblogs, replies,
                compound_score, pos_score,
                neg_score, neu_score,
                sentiment_category, collection_method
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (post_id) DO NOTHING
        """, (
            row['post_id'],
            row['topic'],
            row['hashtag'],
            row['created_at'],
            row['content'],
            row['clean_content'],
            row['favourites'],
            row['reblogs'],
            row['replies'],
            row['compound_score'],
            row['pos_score'],
            row['neg_score'],
            row['neu_score'],
            row['sentiment_category'],
            'live'
        ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"✅ Inserted {len(df)} rows into Postgres")


# Define tasks
init_db_table_task = PythonOperator(
    task_id='create_table',
    python_callable=create_table,
    dag=dag,
)

collect_task = PythonOperator(
    task_id='collect_data',
    python_callable=collect_data,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    dag=dag,
)

sentiment_task = PythonOperator(
    task_id='analyze_sentiment',
    python_callable=analyze_sentiment,
    dag=dag,
)

store_task = PythonOperator(
    task_id='store_in_database',
    python_callable=store_in_database,
    dag=dag,
)

# Pipeline flow
init_db_table_task >> collect_task >> clean_task >> sentiment_task >> store_task