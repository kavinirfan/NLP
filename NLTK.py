# Tokenization and Text Normalization using NLTK
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
# Download required NLTK data (updated for latest versions of NLTK)
nltk.download(&#39;punkt&#39;)
nltk.download(&#39;punkt_tab&#39;) # NEW requirement in latest NLTK versions
nltk.download(&#39;wordnet&#39;)
nltk.download(&#39;stopwords&#39;)
nltk.download(&#39;omw-1.4&#39;) # For better lemmatization
text = &quot;NLTK is a powerful library for text processing. It helps in tokenization,
stemming, and lemmatization!&quot;
# 1. Sentence Tokenization
sentences = sent_tokenize(text)
# 2. Word Tokenization
words = word_tokenize(text)
# 3. Lowercasing
lower_words = [w.lower() for w in words]
# 4. Remove Stopwords
stop_words = set(stopwords.words(&quot;english&quot;))
filtered_words = [w for w in lower_words if w not in stop_words and w.isalpha()]
# 5. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_words]
# 6. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
# Display Outputs
print(&quot;\nOriginal Text:&quot;)
print(text)
print(&quot;\nSentence Tokenization:&quot;)

print(sentences)
print(&quot;\nWord Tokenization:&quot;)
print(words)
print(&quot;\nLowercase Words:&quot;)
print(lower_words)
print(&quot;\nFiltered Words (Stopwords Removed):&quot;)
print(filtered_words)
print(&quot;\nStemmed Words:&quot;)
print(stemmed_words)
print(&quot;\nLemmatized Words:&quot;)
print(lemmatized_words)
