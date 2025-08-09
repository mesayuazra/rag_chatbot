from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity, answer_correctness
import pandas as pd
from ragas import evaluate
import json
from dotenv import load_dotenv
from openai import OpenAI
import os
import matplotlib.pyplot as plt
import streamlit as st

load_dotenv()
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

#load dataset
with open('data/dataset.json', 'r', encoding='utf-8') as f:
  raw_data = json.load(f)
  
#convert to dataframe to hf dataset
df = pd.DataFrame(raw_data)
df.rename(columns={'contexts':'retrieved_contexts'}, inplace=True)
dataset = Dataset.from_pandas(df)

#choose metrics
metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity, answer_correctness]

#evaluate
result = evaluate(dataset, metrics)

#result show
print('Evaluation result:')
print(result)

# #visualitation ragas result
# metric_names = [metric['name'] for metric in result.scores]
# scores = [metric['score'] for metric in result.scores]

# plt.figure(figsize=(10,6))
# plt.bar(metrics, scores, color="skyblue")
# plt.ylim(0, 1)
# plt.title("Hasil Evaluasi Chatbot dengan RAGAS")
# plt.ylabel("Skor (0 - 1)")
# plt.xticks(rotation=30)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()