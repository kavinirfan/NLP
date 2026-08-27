import nltk
from nltk import CFG
from nltk.parse import ChartParser

# Define grammar
grammar = CFG.fromstring(&quot;&quot;&quot;
S -&gt; NP VP
NP -&gt; Det N | Det Adj N
VP -&gt; V NP
Det -&gt; &#39;the&#39; | &#39;a&#39;
N -&gt; &#39;cat&#39; | &#39;dog&#39;
Adj -&gt; &#39;big&#39;
V -&gt; &#39;chased&#39; | &#39;saw&#39;

&quot;&quot;&quot;)

# Create parser
parser = ChartParser(grammar)

# Sentence to parse
sentence = &quot;the big dog chased the cat&quot;.split()

# Parse sentence
for tree in parser.parse(sentence):
print(tree)
tree.pretty_print()
