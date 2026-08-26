import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Download required NLTK datasets
nltk.download(&#39;punkt&#39;)
nltk.download(&#39;punkt_tab&#39;) # required for newer NLTK versions
nltk.download(&#39;wordnet&#39;)
nltk.download(&#39;omw-1.4&#39;)
# Sample text
text = &quot;The striped bats are hanging on their feet for best.&quot;
# Tokenize words
words = word_tokenize(text)
# Initialize stemmers and lemmatizer
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
# Apply stemming
porter_stems = [porter.stem(w) for w in words]
lancaster_stems = [lancaster.stem(w) for w in words]
# Apply lemmatization
lemmas = [lemmatizer.lemmatize(w) for w in words]
# Output results
print(&quot;\nOriginal Text:&quot;)
print(text)
print(&quot;\nTokenized Words:&quot;)
print(words)
print(&quot;\nPorter Stemming:&quot;)
print(porter_stems)
print(&quot;\nLancaster Stemming:&quot;)
print(lancaster_stems)
print(&quot;\nLemmatization:&quot;)
print(lemmas)
