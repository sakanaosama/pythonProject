import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random


# Load the CSV file to check its content
rules_df = pd.read_csv('./src/table_org.csv')
print(rules_df.head())

rules_df['Tag'] = rules_df[['Tag 1', 'Tag 2', 'Tag 3']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

rules_df['Rules'] = rules_df[['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Move the 'Tag' column to the first position
columns = ['Tag'] + [col for col in rules_df.columns if col != 'Tag']
rules_df = rules_df[columns]






# Drop the original Tag 1, Tag 2, Tag 3 columns
combined_df = rules_df.drop(columns=['Tag 1', 'Tag 2', 'Tag 3', 'Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5', "Your Comments"])






results_csv_path = 'src/table_rules_split-' + f'-{random.randint(1, 999)}.csv'
combined_df.to_csv(results_csv_path, index=False)





print(combined_df)
