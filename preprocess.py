import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Keep required columns
    df = df[['reviewID', 'title', 'rating', 'helpfulness_score']].copy()

    # Handle missing values
    df['title'] = df['title'].fillna('')
    df['rating'] = df['rating'].fillna(df['rating'].median())
    df['helpfulness_score'] = df['helpfulness_score'].fillna(0)

    # Create unique index for ML
    df.reset_index(drop=True, inplace=True)
    df['review_unique_id'] = df.index

    # Normalize numerical features
    scaler = MinMaxScaler()
    df[['rating_norm', 'helpfulness_norm']] = scaler.fit_transform(
        df[['rating', 'helpfulness_score']]
    )

    # TF-IDF for content-based filtering
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['title'])

    return df, tfidf_matrix
