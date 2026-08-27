!pip install spacy
!python -m spacy download en_core_web_sm

# Import spaCy
import spacy

# Load English model
nlp = spacy.load(&quot;en_core_web_sm&quot;)

# Sample text
text = &quot;&quot;&quot;
Sundar Pichai is the CEO of Google.
He was born in India and studied at Stanford University.
Microsoft Corporation is headquartered in Redmond.
&quot;&quot;&quot;

# Process text
doc = nlp(text)

# Print named entities
print(&quot;Named Entities:\n&quot;)

for ent in doc.ents:
    print(f&quot;Text: {ent.text}&quot;)
    print(f&quot;Label: {ent.label_}&quot;)
    print()
