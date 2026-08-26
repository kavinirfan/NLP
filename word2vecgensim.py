# Import required libraries
from gensim.models import Word2Vec

# Sample corpus (tokenized sentences)
sentences = [
[&quot;data&quot;, &quot;science&quot;, &quot;is&quot;, &quot;fun&quot;],
[&quot;machine&quot;, &quot;learning&quot;, &quot;is&quot;, &quot;powerful&quot;],
[&quot;deep&quot;, &quot;learning&quot;, &quot;is&quot;, &quot;part&quot;, &quot;of&quot;, &quot;data&quot;, &quot;science&quot;],
[&quot;i&quot;, &quot;love&quot;, &quot;machine&quot;, &quot;learning&quot;]
]

# Train Word2Vec model
model = Word2Vec(
sentences=sentences,
vector_size=50, # Dimension of word vectors
window=3, # Context window size
min_count=1, # Minimum word frequency
workers=4 # Parallel processing
)

# Get vector for a word
vector = model.wv[&quot;data&quot;]
print(&quot;Vector for the word &#39;data&#39;:\n&quot;)
print(vector)

# Find similar words
similar_words = model.wv.most_similar(&quot;learning&quot;, topn=3)
print(&quot;\nWords similar to &#39;learning&#39;:\n&quot;)
for word, score in similar_words:
print(f&quot;{word} : {score}&quot;)
