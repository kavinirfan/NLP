from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
&quot;I love data science&quot;,
&quot;Data science is fun&quot;,
&quot;I love machine learning&quot;
]

# Create TF-IDF model
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Convert to DataFrame
tfidf_df = pd.DataFrame(
tfidf_matrix.toarray(),
columns=tfidf_vectorizer.get_feature_names_out()
)

print(&quot;TF-IDF Representation:\n&quot;)
print(tfidf_df.round(3))
