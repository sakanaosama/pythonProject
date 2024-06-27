import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

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


# Load the CSV file to check its content
rules_df = pd.read_csv('src/table_rules_3Lv.csv')
print(rules_df.head())

with open('./src/target_Animal.txt', 'r', encoding='utf-8') as file:
    document_text = file.read()

segments = split_text(document_text, 100, 50)

# Vectorize the rules and segments
vectorizer = TfidfVectorizer()
rules_vector = vectorizer.fit_transform(rules_df['Rules'])
segments_vector = vectorizer.transform(segments)

# Calculate cosine similarity for each segment against all rules
similarity_scores = cosine_similarity(segments_vector, rules_vector)

# Find the highest scoring segment for each rule
best_segments_indices = np.argmax(similarity_scores, axis=0)
best_scores = np.max(similarity_scores, axis=0)

# Extracting the best segment and tag information
best_segments = [segments[idx] for idx in best_segments_indices]
results = pd.DataFrame({
    'Tag1': rules_df['Tag'],
    # 'Tag2': rules_df['Tag 2'],
    'Score': best_scores,
    'Best Segment': best_segments
})

print(results.head())

second_best_indices = np.argsort(similarity_scores, axis=0)[-2]
second_best_segments = [segments[idx] for idx in second_best_indices]

# Update the DataFrame to include the second best segments
results['2nd Best Segment'] = second_best_segments

print(results.head())
# Save the updated results to a CSV file
results_csv_path = 'src/tst-rst-3Lv-withBest Segment-' + f'-{random.randint(1, 999)}.csv'
results.to_csv(results_csv_path, index=False)

print(results_csv_path)


# part2(results_csv_path)
