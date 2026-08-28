from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# --------------------------------
# 1. Text Generation using GPT-2
# --------------------------------
generator = pipeline(&quot;text-generation&quot;, model=&quot;gpt2&quot;)
result = generator(&quot;Artificial Intelligence&quot;, max_length=20)
print(&quot;Generated Text:&quot;)
print(result[0][&#39;generated_text&#39;])
# --------------------------------
# 2. Chatbot (Question Answering)
# --------------------------------
qa = pipeline(&quot;question-answering&quot;)
context = &quot;Artificial Intelligence helps machines learn from data.&quot;
question = &quot;What helps machines learn?&quot;
answer = qa(question=question, context=context)
print(&quot;\nChatbot Answer:&quot;)
print(answer[&#39;answer&#39;])
# --------------------------------
# 3. Machine Translation
# --------------------------------
model_name = &quot;Helsinki-NLP/opus-mt-en-fr&quot;
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
text = &quot;Machine learning is powerful&quot;
inputs = tokenizer(text, return_tensors=&quot;pt&quot;)
outputs = model.generate(**inputs)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(&quot;\nTranslation:&quot;)
print(translation)
# --------------------------------
# 4. Speech Recognition
# --------------------------------
speech = pipeline(&quot;automatic-speech-recognition&quot;,
                  model=&quot;openai/whisper-base&quot;)
print(&quot;\nSpeech Recognition model loaded successfully&quot;)
