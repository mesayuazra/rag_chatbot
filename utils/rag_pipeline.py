import os
import numpy as np
import faiss
import fitz
import re
# from nltk.tokenize import sent_tokenize
from openai import OpenAI
from dotenv import load_dotenv
import json
import streamlit as st
import shutil

load_dotenv()
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

CHUNK_FILE = 'data/chunks.json'
FAISS_INDEX_FILE = 'data/faiss.index'

os.makedirs('data', exist_ok=True)

class RAGPipeline:
  def __init__(self):
    # self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    self.index = None #faiss index to be created later
    self.chunks = {} #store doc chunks for retrieval
  
  #load the pdf
  def load_pdf(self, file_path):
    text = ""
    with fitz.open(file_path) as doc:
      for page in doc:
        text += page.get_text('text') + '\n' #if using line breaks
        # clean_text = re.sub(r'\s',' ',text).strip() #cleaning the white spaces
    #if using line breaks
    text = text.replace('\t', ' ')
    lines = [line.strip() for line in text.split('\n')]
    clean_text = '\n'.join(lines) 
    return clean_text.strip() #clean_text 
  
  #chunking the text
  def chunk_text(self, text, filename):
    chunks = chunk_text_by_marker(text)
    self.chunks[filename] = chunks
    return chunks
  
  #build faiss index
  def build_faiss_index(self):
    all_chunks = []
    for chunk_list in self.chunks.values():
      all_chunks.extend(chunk_list)
      
    if not all_chunks:
      return
    
    print(f'Embedding {len(all_chunks)} chunks...')
    embeddings = [get_embedding(chunk) for chunk in all_chunks]
    matrix = np.array(embeddings).astype('float32') #convert to proper format for faiss
    self.index = faiss.IndexFlatL2(matrix.shape[1]) #cretae index with l2 distance
    self.index.add(matrix) #add them to the faiss index
  
  #retrieve the chunks with search index similarity   
  def retrieve_chunks(self, query, top_k=2):
    all_chunks = []
    for chunk_list in self.chunks.values():
      all_chunks.extend(chunk_list)

    if not self.index or not all_chunks:
        return []
      
    response = client.embeddings.create(
      model='text-embedding-3-small',
      input=[query]
    )
    q_vector = np.array(response.data[0].embedding).astype('float32').reshape(1,-1) #embed the query (user's question) into a vector
    distances, indices = self.index.search(q_vector, top_k) #search the closest vector (top-k similar chunks)
    return [all_chunks[i] for i in indices[0]] #return the retrieved chunks

#chunking   
def chunk_text_by_marker(text, marker='•', group_size=2):
  lines = text.split('\n')
  chunks = []
  current_chunk = []
  
  # building sections
  for line in lines:
    line = line.strip()
    if not line:
      continue
    
    if line.startswith(marker):
      if current_chunk:
        chunks.append('\n'.join(current_chunk)) 
        current_chunk = []
    current_chunk.append(line)  
    
  if current_chunk:
    chunks.append('\n'.join(current_chunk))
    
  chunks = []
  for i in range(0, len(chunks), group_size):
    chunk = "\n\n".join(chunks[i:i+group_size]) #adding blank line
    chunks.append(chunk)
      
  return chunks
  
def load_all_pdfs_and_index(rag: RAGPipeline, folder_path="uploads"):
  if not os.path.exists(folder_path):
    return

  for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
      filepath = os.path.join(folder_path, filename)
      text = rag.load_pdf(filepath)
      rag.chunk_text(text, filename)

  if rag.chunks:
    print(f"Total chunks: {len(rag.chunks)}")
    rag.build_faiss_index()

def get_embedding(text, model='text-embedding-3-small'):
  response = client.embeddings.create(input=[text], model=model)
  return response.data[0].embedding

INDEXED_RECORD_PATH = 'indexed.json'

def get_indexed_files():
  if os.path.exists(INDEXED_RECORD_PATH):
    with open(INDEXED_RECORD_PATH, 'r') as f:
      return set(json.load(f))
  return set()

#marked it as an indexed file
def mark_file_as_indexed(filename):
  files = get_indexed_files()
  files.add(filename)
  with open(INDEXED_RECORD_PATH, 'w') as f:
    json.dump(list(files), f)
    
#load chunks
def load_chunks(filepath=CHUNK_FILE):
    if os.path.isdir(filepath):
      shutil.rmtree(filepath)  # remove the folder
      os.makedirs(os.path.dirname(filepath), exist_ok=True)  # ensure "data" exists
      with open(filepath, "w", encoding="utf-8") as f:
        f.write("{}")  # empty JSON object

    # ✅ If file doesn't exist, return empty dict
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
      return json.load(f)
    
#saving the chunks
def save_chunks(chunks_dict, filepath=CHUNK_FILE):
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(chunks_dict, f, ensure_ascii=False, indent=2)
      
def delete_chunks(pdf_filename, filepath='data/chunks.json'):
  chunks_dict = load_chunks()
  if pdf_filename in chunks_dict:
    del chunks_dict[pdf_filename]
    save_chunks(chunks_dict, filepath)

def save_faiss_index(index, filepath=FAISS_INDEX_FILE):
    faiss.write_index(index, filepath)

def load_faiss_index(filepath=FAISS_INDEX_FILE):
    if not os.path.exists(filepath):
        return None
    return faiss.read_index(filepath)