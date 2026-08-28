Program:
# Install transformers library if not installed
# !pip install transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# Load Seq2Seq translation model directly
tokenizer = AutoTokenizer.from_pretrained(&quot;Helsinki-NLP/opus-mt-en-fr&quot;)
model_translation = AutoModelForSeq2SeqLM.from_pretrained(&quot;Helsinki-NLP/opus-mt-
en-fr&quot;)
# Input sentence
text = &quot;Natural Language Processing is an important field of Artificial Intelligence.&quot;
# Tokenize and generate translation

inputs = tokenizer(text, return_tensors=&quot;pt&quot;, padding=True, truncation=True)
translated_ids = model_translation.generate(**inputs)
# Decode the translated text
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
# Print output
print(&quot;Original Text:&quot;, text)
print(&quot;Translated Text:&quot;, translated_text)
