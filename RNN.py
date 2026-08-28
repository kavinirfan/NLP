import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
import numpy as np # Import numpy
print(&quot;RNN TEXT CLASSIFICATION&quot;)
texts = [&quot;I love AI&quot;, &quot;I hate AI&quot;]
labels = [1,0]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
seq = tokenizer.texts_to_sequences(texts)
seq = pad_sequences(seq)
# Explicitly convert to TensorFlow Tensors for model.fit
seq_tensor = tf.constant(seq, dtype=tf.int32)
labels_tensor = tf.constant(labels, dtype=tf.float32) # Labels should be float for
# binary_crossentropy with sigmoid output
model = Sequential([
Embedding(len(tokenizer.word_index) + 1, 8), # Adjusted input_dim to be precise based
# on vocabulary size
SimpleRNN(8),
Dense(1,activation=&quot;sigmoid&quot;)
])

model.compile(optimizer=&quot;adam&quot;,loss=&quot;binary_crossentropy&quot;)
model.fit(seq_tensor, labels_tensor, epochs=3) # Use the TensorFlow tensors
print(&quot;\nBERT MODEL&quot;)
bert = pipeline(&quot;sentiment-analysis&quot;)
print(bert(&quot;NLP is interesting&quot;))
print(&quot;\nGPT TEXT GENERATION&quot;)
gpt = pipeline(&quot;text-generation&quot;,model=&quot;gpt2&quot;)
print(gpt(&quot;AI is&quot;,max_length=15))
print(&quot;\nSEQUENCE TO SEQUENCE&quot;)
text=&quot;hello&quot;
print(&quot;input:&quot;,text)
print(&quot;output:&quot;,text[::-1])
print(&quot;\nTRANSFER LEARNING&quot;)
print(&quot;Using pretrained BERT model&quot;)
