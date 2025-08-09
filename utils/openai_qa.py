import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
client = OpenAI()
client.api_key = st.secrets['OPENAI_API_KEY']

#build prompt to guide the llm's generate answer, combine top chunks with user query for context
def create_prompt(query, chunks):
  context = '\n\n'.join(chunks)
  prompt = f'''Kamu adalah asisten cerdas. Jawab pertanyaan pengguna dengan lengkap hanya berdasarkan konteks di bawah ini. 
  Namun, jika jawaban tidak ada di dalam konteks, jawab 'mohon maaf saya tidak tahu mengenai informasi tersebut. Silahkan cek situs resmi Gunadarma untuk informasi lebih lanjut.'\n\n
Context:\n{context}\n\n 
Question: {query}\n
Answer:
'''
  return prompt

#sends prompt to openai's api and retrieve response
def ask_openai(prompt):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{'role':'user', 'content':prompt}],
    temperature=0.2,
    max_tokens=500,
    stream=True,
  )
  return response 