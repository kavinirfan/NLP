from transformers import (pipeline, AutoTokenizer,

AutoModelForSeq2SeqLM)
# -------------------------------
# 1. Basic Chatbot
# -------------------------------
print(&quot;Simple Chatbot (type &#39;bye&#39; to exit)&quot;)
responses = {
&quot;hello&quot;: &quot;Hi! How can I help you?&quot;,
&quot;how are you&quot;: &quot;I am fine. How can I assist you?&quot;,

&quot;what is ai&quot;: &quot;AI means Artificial Intelligence.&quot;
}
user_input = &quot;hello&quot;
if user_input.lower() in responses:
    print(&quot;Chatbot:&quot;, responses[user_input.lower()])
else:
    print(&quot;Chatbot: I don&#39;t understand.&quot;)
# -------------------------------
# 2. Text Summarization
# -------------------------------
# Manually load tokenizer and model for summarization
# since &#39;summarization&#39; task is not recognized by pipeline in this environment
tokenizer = AutoTokenizer.from_pretrained(&quot;sshleifer/distilbart-cnn-12-6&quot;)
model_summarization =
AutoModelForSeq2SeqLM.from_pretrained(&quot;sshleifer/distilbart-cnn-12-6&quot;)
text = &quot;&quot;&quot;
Artificial Intelligence is transforming modern technology.
It helps machines learn from data and make intelligent decisions.
AI is widely used in healthcare, education, and business.
&quot;&quot;&quot;
# Tokenize and generate summary
inputs = tokenizer([text], max_length=1024, return_tensors=&#39;pt&#39;,
truncation=True)
summary_ids = model_summarization.generate(
inputs[&#39;input_ids&#39;], num_beams=4, max_length=30, min_length=10,
early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(&quot;\nSummary:&quot;)
print(summary)
# -------------------------------
# 3. Question Answering System
# -------------------------------
qa = pipeline(&quot;question-answering&quot;)
context = &quot;Artificial Intelligence helps machines learn from data.&quot;
question = &quot;What helps machines learn?&quot;
answer = qa(question=question, context=context)
print(&quot;\nAnswer:&quot;)
print(answer[&#39;answer&#39;])
# -------------------------------

# 4. Information Retrieval
# -------------------------------
documents = [
&quot;NLP helps computers understand human language&quot;,
&quot;Machine learning improves prediction systems&quot;,
&quot;Artificial Intelligence is used in healthcare&quot;
]
query = &quot;language&quot;
results = [doc for doc in documents if query in doc.lower()]
print(&quot;\nInformation Retrieval Result:&quot;)
for r in results:
    print(r)
