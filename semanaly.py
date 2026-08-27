# Step 1: Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Step 2: Sample documents
documents = [
&quot;I love machine learning and data science&quot;,
&quot;Artificial intelligence and machine learning are related fields&quot;,
&quot;Big data analytics and data mining&quot;,
&quot;Football and cricket are popular sports&quot;,
&quot;Tennis and badminton are indoor sports&quot;,
&quot;Sports events include football and cricket matches&quot;
]

# Step 3: Convert text to TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words=&#39;english&#39;)
X = vectorizer.fit_transform(documents)

# Step 4: Apply LSA (Truncated SVD)
svd = TruncatedSVD(n_components=2, random_state=42)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X_lsa = lsa.fit_transform(X)

# Step 5: Apply KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_lsa)

# Step 6: Print cluster results
print(&quot;Cluster assignments:\n&quot;)

for i, doc in enumerate(documents):
print(f&quot;Document {i+1}: Cluster {kmeans.labels_[i]}&quot;)
print(&quot;Text:&quot;, doc)
print()
