# Sentiment Analysis using Naive Bayes

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Training data
texts = [
&quot;I love this product&quot;,
&quot;This is an amazing movie&quot;,
&quot;I hate this item&quot;,
&quot;This is a bad product&quot;
]

labels = [&quot;positive&quot;,&quot;positive&quot;,&quot;negative&quot;,&quot;negative&quot;]

# Create model
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train model
model.fit(texts, labels)

# Test sentence
test_text = [&quot;This movie is amazing&quot;]

prediction = model.predict(test_text)

print(&quot;Text:&quot;, test_text[0])
print(&quot;Sentiment:&quot;, prediction[0])

# -----------------------------------
# Deep Learning Models for NLP
# -----------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense

nn_model = Sequential()

# Embedding layer
nn_model.add(Embedding(input_dim=1000, output_dim=64))

# RNN, LSTM, GRU layers
nn_model.add(SimpleRNN(32))
nn_model.add(LSTM(32))
nn_model.add(GRU(32))

# Output layer

nn_model.add(Dense(1, activation=&#39;sigmoid&#39;))

print(&quot;\nNeural Network model created successfully&quot;)
