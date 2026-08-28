# Install transformers if needed
# !pip install transformers

from transformers import pipeline

# Load Question Answering pipeline
qa = pipeline(&quot;question-answering&quot;)

# Context paragraph
context = &quot;&quot;&quot;
Artificial Intelligence (AI) is a branch of computer science that aims to create
machines capable of intelligent behavior. AI is used in applications such as
speech recognition, machine translation, and recommendation systems.
&quot;&quot;&quot;

# Question
question = &quot;What does AI aim to create?&quot;

# Get answer
result = qa(question=question, context=context)

# Print result
print(&quot;Question:&quot;, question)
print(&quot;Answer:&quot;, result[&#39;answer&#39;])
