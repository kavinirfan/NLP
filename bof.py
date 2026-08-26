from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample documents
documents = [
&quot;I love data science&quot;,
&quot;Data science is fun&quot;,
&quot;I love machine learning&quot;
]
# Create Bag-of-Words model
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)
# Convert to DataFrame for better display
bow_df = pd.DataFrame(
bow_matrix.toarray(),
columns=vectorizer.get_feature_names_out()
)
print(&quot;Bag-of-Words Representation:\n&quot;)
print(bow_df)
