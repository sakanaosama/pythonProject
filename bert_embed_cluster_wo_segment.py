import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

import numpy as np
import random
import torch



def split_text(text, segment_size, overlap):
    words = text.split()
    segments = []
    for start in range(0, len(words), segment_size - overlap):
        segment = " ".join(words[start:start + segment_size])
        segments.append(segment)
    return segments

def part2(results_csv_path):
    # part 2: transform csv
    uploaded_file_path = results_csv_path

    # Load the uploaded CSV file
    data_uploaded = pd.read_csv(uploaded_file_path)

    # Splitting the data based on 'Tag1' and saving into separate files
    tags_uploaded = data_uploaded['Tag1'].unique()  # Extract unique tags
    output_files_uploaded = {}

    for tag in tags_uploaded:
        # Filter data for the current tag
        filtered_data = data_uploaded[data_uploaded['Tag1'] == tag]

        data_without_columns = filtered_data.iloc[:, 3:]

        best_segments = data_without_columns['Best Segment'].tolist()
        second_best_segments = data_without_columns['2nd Best Segment'].tolist()

        best_segments = list(set(best_segments))
        second_best_segments = list(set(second_best_segments))

        # Combine both lists
        all_segments = best_segments + second_best_segments

        # Remove redundancy by converting the list to a set and then back to a list
        unique_segments = list(set(all_segments))

        # Print the unique segments
        for segment in unique_segments:
            print(segment)

        combined_segments = ' '.join(unique_segments)

        # Extract the tag number assuming tag format '1. Organizational Management'
        tag_number = tag.split('.')[0].strip()

        # Generate a file name for the current tag

        output_file_path = f'src/sub/test-rst_{tag_number}_puretext.txt'
        with open(output_file_path, 'w') as file:
            file.write(combined_segments)

        file_name_uploaded = f'src/sub/test-rst_{tag_number}_organizational_management.csv'

        # Save filtered data to a new CSV file
        data_without_columns.to_csv(file_name_uploaded, index=False)

        # Store the file name for reference
        output_files_uploaded[tag] = file_name_uploaded

    print(output_files_uploaded)

print(np.__version__)  # Should print a version lower than 2.0.0

# 1. Load the CSV file to check its content
rules_df = pd.read_csv('src/table_rules_3Lv.csv')
with open('./src/target_Animal.txt', 'r', encoding='utf-8') as file:
    document_text = file.read()

print(rules_df.head())
print(document_text[:500])

# 2. Segment the Document:
segments = split_text(document_text, 500, 100)


# 3. Load BERT Model and Tokenizer:
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


rule_embeddings = [encode_text(rule, tokenizer, model) for rule in rules_df['Rules']]
segment_embeddings = [encode_text(segment, tokenizer, model) for segment in segments]


# 5. Cluster Segments:
from sklearn.cluster import KMeans
# n_samples = len(segment_embeddings)
n_samples = 506
n_clusters = 506
# n_clusters = max(len(rules_df), n_samples)  # Choose an appropriate number of clusters


# n_clusters = len(rules_df)
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
# kmeans.fit(segment_embeddings)
kmeans.fit(506)
clusters = kmeans.predict(segment_embeddings)

# 6. Find Best Matching Segments for Each Rule:
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# def find_best_segments(rule_embeddings, segment_embeddings, clusters, n_clusters):
#     best_segments = []
#     best_scores = []
#     for i in range(n_clusters):
#         cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
#         cluster_embeddings = [segment_embeddings[index] for index in cluster_indices]
#         similarities = cosine_similarity([rule_embeddings[i]], cluster_embeddings)[0]
#         best_index = cluster_indices[np.argmax(similarities)]
#         best_segments.append(segments[best_index])
#         best_scores.append(np.max(similarities))
#     return best_segments, best_scores
#
# best_segments, best_scores = find_best_segments(rule_embeddings, segment_embeddings, clusters, n_clusters)


def find_best_scores(rule_embeddings, segment_embeddings, clusters, n_clusters):
    best_scores = []
    for i in range(n_clusters):
        cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
        cluster_embeddings = [segment_embeddings[index] for index in cluster_indices]
        similarities = cosine_similarity([rule_embeddings[i]], cluster_embeddings)[0]
        best_scores.append(np.max(similarities))
    return best_scores


best_scores = find_best_scores(rule_embeddings, segment_embeddings, clusters, n_clusters)



# 7. Compile Results:
results = pd.DataFrame({
    'Tag1': rules_df['Tag'][:n_clusters],
    'Score': best_scores
})
print(results.head())




# Save Results to CSV:
import random

results_csv_path = 'src/tst-rst-3Lv-withBest Segment-bert-' + f'-{random.randint(1, 999)}.csv'
results.to_csv(results_csv_path, index=False)
print(results_csv_path)
