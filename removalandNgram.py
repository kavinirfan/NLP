from sklearn.feature_extraction.text import CountVectorizer
# Sample documents
docs = [
&quot;Data science is an interdisciplinary field&quot;,
&quot;Science involves experiments and data&quot;,
&quot;Data is the new oil&quot;
]
# Create CountVectorizer with stop word removal and N-gram generation (bigrams)
vectorizer = CountVectorizer(stop_words=&#39;english&#39;, ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)
# Display feature names and matrix
feature_names = vectorizer.get_feature_names_out()
matrix = X.toarray()
feature_names, matrix
