import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use a different sentence tokenizer due to issues with NLTK download
from sklearn.feature_extraction.text import CountVectorizer
# import ace_tools as tools; tools.display_dataframe_to_user(name="Relevance Evaluation Results", dataframe=results_df)


# Function to split text into segments of 500 words with 100 words overlap
def split_text(text, segment_size=500, overlap=100):
    words = text.split()
    segments = []
    for start in range(0, len(words), segment_size - overlap):
        segment = " ".join(words[start:start + segment_size])
        segments.append(segment)
    return segments

# Load the table_rules_2Lv.csv file and the target.txt file
rules_df = pd.read_csv('src/table_rules_2Lv.csv')

with open('./src/target_Animal.txt', 'r', encoding='utf-8') as file:
    target_text = file.read()

# Display the rules dataframe and the first 500 characters of the target text for an initial look
# rules_df, target_text

print(rules_df.head())
print(target_text[:500])

#
#
rules = rules_df['Rules'].tolist()

# # Use simple split method for sentence tokenization
# target_segments = target_text.split('\n')

# Split the target text into segments
target_segments = split_text(target_text)



#
# Create a TfidfVectorizer and fit it on the combined rules and target text segments
vectorizer = TfidfVectorizer().fit(rules + target_segments)
#
# Transform the rules and target text segments into TF-IDF vectors
rules_vectors = vectorizer.transform(rules)
target_vectors = vectorizer.transform(target_segments)
#
# Compute cosine similarity between each rule and each target text segment
similarity_matrix = cosine_similarity(rules_vectors, target_vectors)

# Find the highest similarity score for each rule and corresponding target text segment
similarity_scores = similarity_matrix.max(axis=1)
similarity_segments = [target_segments[similarity_matrix[i].argmax()] for i in range(len(rules))]

# Combine tags, rules, scores, and target text segments into a dataframe
results_df = pd.DataFrame({
    'Tag1': rules_df['Tag 1'],
    'Tag2': rules_df['Tag 2'],
    'Rule': rules,
    'Score': similarity_scores,
    'Target Text Segment': similarity_segments
})

print(results_df)
