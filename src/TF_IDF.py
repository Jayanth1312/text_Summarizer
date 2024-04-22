from sklearn.feature_extraction.text import TfidfVectorizer
from dataCleaning import data_cleaning
import pandas as pd
import numpy as np

def read_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        documents = file.readlines()
    return documents

file_path = 'data/training/test1.txt'

raw_documents = read_documents(file_path)
documents = [data_cleaning(doc) for doc in raw_documents]

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

import numpy as np

sentence_scores = np.sum(df_tfidf, axis=1).values

N = 2
top_sentence_indices = np.argsort(-sentence_scores)[:N]

summary_sentences = [documents[idx].strip() for idx in top_sentence_indices]


summary = ' '.join(summary_sentences)
print("Summary:\n", summary)
