from transformers import pipeline
from sklearn.metrics import accuracy_score
# -----------------------------
# 1. Fine-Tuning Pretrained Model (Sentiment Analysis using BERT)
# -----------------------------
sentiment_model = pipeline(&quot;sentiment-analysis&quot;)
text = &quot;This NLP lab is very easy and interesting&quot;
result = sentiment_model(text)
print(&quot;Input Text:&quot;, text)
print(&quot;Sentiment Result:&quot;, result)
# -----------------------------
# 2. Text Generation using Deep Learning Model
# -----------------------------
generator = pipeline(&quot;text-generation&quot;, model=&quot;gpt2&quot;)
generated = generator(&quot;Deep Learning&quot;, max_length=25)
print(&quot;\nGenerated Text:&quot;)
print(generated[0][&#39;generated_text&#39;])
# -----------------------------
# 3. Evaluating Deep Learning Model for NLP
# -----------------------------
y_true = [1,0,1,1,0]
y_pred = [1,0,1,0,0]
accuracy = accuracy_score(y_true, y_pred)
print(&quot;\nModel Accuracy:&quot;, accuracy)
