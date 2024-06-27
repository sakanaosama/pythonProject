import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
print(np.__version__)  # Should print a version lower than 2.0.0

# 1. Load the Rules and Document:
rules_df = pd.read_csv('src/table_rules_3Lv.csv')
with open('./src/target_Animal.txt', 'r', encoding='utf-8') as file:
    document_text = file.read()

print(rules_df.head())
print(document_text[:500])


def split_text(text, segment_size, overlap):
    words = text.split()
    segments = []
    for start in range(0, len(words), segment_size - overlap):
        segment = " ".join(words[start:start + segment_size])
        segments.append(segment)
    return segments


segments = split_text(document_text, 500, 100)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

rule_embeddings = [encode_text(rule, tokenizer, model) for rule in rules_df['Rules']]
segment_embeddings = [encode_text(segment, tokenizer, model) for segment in segments]

n_samples = len(segment_embeddings)
n_clusters = min(len(rules_df), n_samples)

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(segment_embeddings)
clusters = kmeans.predict(segment_embeddings)

def find_best_segments(rule_embeddings, segment_embeddings, clusters, n_clusters):
    best_segments = []
    best_scores = []
    for i in range(n_clusters):
        cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
        cluster_embeddings = [segment_embeddings[index] for index in cluster_indices]
        similarities = cosine_similarity([rule_embeddings[i]], cluster_embeddings)[0]
        best_index = cluster_indices[np.argmax(similarities)]
        best_segments.append(segments[best_index])
        best_scores.append(np.max(similarities))
    return best_segments, best_scores

best_segments, best_scores = find_best_segments(rule_embeddings, segment_embeddings, clusters, n_clusters)

results = pd.DataFrame({
    'Tag1': rules_df['Tag'][:n_clusters],  # Ensure the lengths match
    'Score': best_scores,
    'Best Segment': best_segments
})
print(results.head())

def find_second_best_segments(rule_embeddings, segment_embeddings, clusters, n_clusters):
    second_best_segments = []
    for i in range(n_clusters):
        cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
        cluster_embeddings = [segment_embeddings[index] for index in cluster_indices]
        similarities = cosine_similarity([rule_embeddings[i]], cluster_embeddings)[0]
        if len(similarities) > 1:
            sorted_indices = np.argsort(similarities)[-2]
        else:
            sorted_indices = np.argmax(similarities)
        second_best_segments.append(segments[cluster_indices[sorted_indices]])
    return second_best_segments

second_best_segments = find_second_best_segments(rule_embeddings, segment_embeddings, clusters, n_clusters)
results['2nd Best Segment'] = second_best_segments
print(results.head())

results_csv_path = 'src/tst-rst-3Lv-withBest-Segment-' + f'{random.randint(1, 999)}.csv'
results.to_csv(results_csv_path, index=False)
print(results_csv_path)
