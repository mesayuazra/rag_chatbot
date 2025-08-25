import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from utils.rag_pipeline import RAGPipeline
from utils.openai_qa import create_prompt, ask_openai
from utils.auth import login, admin_register, load_admin
# from utils.history import save_to_history, load_history
from utils.rag_pipeline import get_embedding
from utils.rag_pipeline import load_all_pdfs_and_index
from utils.rag_pipeline import get_indexed_files, mark_file_as_indexed
from utils.rag_pipeline import load_chunks, load_faiss_index
from utils.rag_pipeline import save_chunks, save_faiss_index, delete_chunks
# from pathlib import Path
import json
import numpy as np
import faiss
import base64
import shutil

load_dotenv()
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

st.set_page_config(page_title='🤖 UG Chatbot', layout='wide')

#create folder if not exist
os.makedirs('uploads', exist_ok=True)
  
#initialize session states
if 'logged_in' not in st.session_state:
  st.session_state.logged_in = False
if 'rag' not in st.session_state:
  st.session_state.rag = RAGPipeline()
if 'page' not in st.session_state:
  st.session_state.page = 'chat'
if 'mode' not in st.session_state:
  st.session_state.mode = 'chat'
if 'chat_history' not in st.session_state:
  st.session_state.chat_history = []
if 'admin_page' not in st.session_state:
  st.session_state.admin_page = 'main'
if 'edit_file' not in st.session_state:
  st.session_state.edit_file = None
if 'delete_file' not in st.session_state:
  st.session_state.delete_file = None
  
rag = st.session_state.rag
rag.chunks = load_chunks()
rag.index = load_faiss_index()
#always rebuild chunks + index

load_all_pdfs_and_index(rag)
  
#------sidebar
with st.sidebar:
  if not st.session_state.logged_in:
    st.sidebar.header('Menu')
    #chat menu  
    if st.button("💬 Chat", type='tertiary'):
      st.session_state.mode = "chat"
      st.session_state.page = "chat"

    #riwayat chat menu      
    if st.button("🕘 Riwayat Chat", type='tertiary'):
      st.session_state.mode = "riwayat chat"
      st.session_state.page = "chat"

    #admin login menu 
    if st.button('🔐 Login', key='admin_login_icon', type='tertiary'):
      st.session_state.page = 'login'

  else:
    st.header('Admin Panel')  
    if st.button("📄 Kelola PDF", type='tertiary', key='kelola_pdf_btn'):
      st.session_state.admin_page = "main"
      st.session_state.page = "login"
      st.session_state.mode = "kelola pdf"

    if st.button("📁 Data", type='tertiary'):
      st.session_state.admin_page = "data"
      st.session_state.page = "login"
      st.session_state.mode = "data"

    if st.button("🚪 Logout", type='tertiary'):
      st.session_state.logged_in = False
      st.session_state.page = "chat"
      st.session_state.mode = "chat"
      st.success("Berhasil logout.")
      st.rerun()
    # if st.button("🚨 Reset Semua Data (hapus uploads & chunks.json)"):
    #   try:
    #     # Hapus folder uploads kalau ada
    #     if os.path.exists("uploads"):
    #         shutil.rmtree("uploads")
        
    #     # Hapus file chunks.json kalau ada
    #     if os.path.exists("data/chunks.json"):
    #         os.remove("data/chunks.json")

    #     st.success("✅ Semua data berhasil direset (uploads & chunks.json).")
    #   except Exception as e:
    #     st.error(f"❌ Gagal reset data: {e}")
        
#----------------admin view
if st.session_state.page == 'login':
  if not st.session_state.logged_in:
    login()
    st.stop()
  else: 
    if st.session_state.admin_page == 'main':
      uploaded_files = st.file_uploader(
        'Silahkan unggah satu file PDF atau lebih', 
        type=['pdf'], 
        accept_multiple_files=True)
    
      if uploaded_files:
        for uploaded_file in uploaded_files:
          filename = uploaded_file.name if hasattr(uploaded_file, 'name') else "uploaded.pdf"
          save_path = os.path.join("uploads", filename)
        
          #save uploaded file
          with open(save_path, 'wb') as f:
            f.write(uploaded_file.read())
          st.success(f'Berhasil diunggah: {filename}')
      
          indexed_files = get_indexed_files()
          if filename not in indexed_files:
            new_text = st.session_state.rag.load_pdf(save_path)
            new_chunks = st.session_state.rag.chunk_text(new_text, filename)

            new_embeddings = [get_embedding(chunk) for chunk in new_chunks]
            new_matrix = np.array(new_embeddings).astype("float32")

            if st.session_state.rag.index is None:
              st.session_state.rag.index = faiss.IndexFlatL2(new_matrix.shape[1])
            st.session_state.rag.index.add(new_matrix)

            chunks_dict = load_chunks()
            chunks_dict[filename] = new_chunks
            save_chunks(chunks_dict)
            mark_file_as_indexed(filename)
          
            st.info("📚 PDF berhasil diproses & ditambahkan ke index.")
          else:
            st.warning("⚠️ File ini sudah pernah diunggah dan diproses.")

      #PDF Manager UI
      st.subheader("🗂️ Kelola PDF")
      #search input
      all_pdf_files = [f for f in os.listdir('uploads') if f.endswith('.pdf')]
      search = st.text_input('🔍 Cari PDF')
      
      #apply filter
      if search:
        pdf_files = [f for f in all_pdf_files if search.lower() in f.lower()]
      else:
        pdf_files = all_pdf_files
      
      #show result  
      if not pdf_files:
        st.warning('Belum ada file yang diunggah.')
      else:
        for pdf_file in pdf_files:
          col1,col2,col3,col4 = st.columns([4,1,1,1])
          with col1:
            st.markdown(f'{pdf_file}')
          #edit file
          with col2:
            if st.button('📖', key=f'read_{pdf_file}'):
              st.session_state.read_file = pdf_file
              st.session_state.admin_page = 'read'
              st.rerun()
          with col3:
            if st.button("✏️", key=f"edit_{pdf_file}"):
              st.session_state.edit_file = pdf_file
              st.session_state.admin_page = 'edit'
              st.rerun()
          #delete file
          with col4:
            if st.button("🗑️", key=f"delete_{pdf_file}"):
              st.session_state.delete_file = pdf_file
              st.session_state.admin_page = 'delete'
              st.rerun()

    elif st.session_state.admin_page == 'read':
      pdf_file = st.session_state.read_file
      pdf_path = os.path.join('uploads', pdf_file)
      
      st.subheader(f'Membaca : {pdf_file}')
      
      with open(pdf_path, 'rb') as f:
        st.download_button(
        label="Download PDF",
        data=f,
        file_name=pdf_file,
        mime="application/pdf"
        )
        
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
        st.components.v1.html(pdf_display, height=610, scrolling=True)
        
      if st.button('Kembali'):
        st.session_state.admin_page = 'main'
        st.session_state.read_file = None
        st.rerun()
        
    elif st.session_state.admin_page == 'edit':
      st.subheader('✏️ Ubah Nama File PDF')
      
      current_file = st.session_state.edit_file
      if current_file:
        current_name = current_file.replace('.pdf', '')
        new_name = st.text_input('Ganti nama file:', value=current_name)
        
        col1, col2, col3 = st.columns([7,1,1])
        
        with col2:
          if st.button('Simpan'):
            old_path = os.path.join('uploads', current_file)
            new_path = os.path.join('uploads', new_name + '.pdf')
            if os.path.exists(new_path):
              st.error('Nama file sudah digunakan.')
            else:
              try:
                os.rename(old_path, new_path)
                st.success(f'Nama file berhasil diubah') 
                
                chunks_dict = load_chunks()
                if current_file in chunks_dict:
                  chunks_dict[new_name] = chunks_dict.pop(current_file)
                  save_chunks(chunks_dict)
                  
                indexed_files = get_indexed_files()
                if current_file in indexed_files:
                  indexed_files.remove(current_file)
                  indexed_files.add(new_name)
                  with open('indexed.json', 'w') as f:
                    json.dump(list(indexed_files), f)
                
                st.session_state.admin_page = 'main'
                st.rerun()
                
              except Exception as e:
                st.error(f'Gagal mengganti nama file: {e}')
                
          with col3:
            if st.button('Batal'):
              st.session_state.admin_page = 'main'
              st.rerun()
    #delete file session          
    elif st.session_state.admin_page == 'delete':
      st.warning(f'Yakin ingin menghapus file: {st.session_state.delete_file}?')
      confirm_col1, confirm_col2 = st.columns(2)
      with confirm_col1:
        if st.button('Hapus'):
          file_to_delete = st.session_state.delete_file
          try:
            file_path = os.path.join('uploads', file_to_delete)
            if os.path.exists(file_path):
              os.remove(file_path)
              # st.success(f"File berhasil dihapus: {file_to_delete}")
            else:
              st.warning(f"File tidak ditemukan: {file_to_delete}")  
            
            #del from chunks.json
            chunks_dict = load_chunks()
            keys_to_delete = [k for k in chunks_dict if os.path.splitext(os.path.basename(k))[0] == os.path.splitext(file_to_delete)[0]]
        
            for key in keys_to_delete:
              del chunks_dict[key]
              # st.info(f"Removed chunks for: {key}")
            save_chunks(chunks_dict)  
                
            #del from indexed.json
            indexed_files = get_indexed_files()
            indexed_files = [f for f in indexed_files if os.path.splitext(os.path.basename(f))[0] != os.path.splitext(file_to_delete)[0]]
            with open('indexed.json', 'w', encoding='utf-8') as f:
              json.dump(list(indexed_files), f)
                    
            #rebuild index
            all_chunks = []
            for chunk_list in chunks_dict.values():
              all_chunks.extend(chunk_list)
              rag = st.session_state.rag
              rag.chunks = chunks_dict
                
            if all_chunks:
              response = client.embeddings.create(
                model='text-embedding-3-small',
                input=all_chunks
                )
              vectors = [np.array(res.embedding) for res in response.data]
              matrix = np.array(vectors).astype('float32')
              rag.index = faiss.IndexFlatL2(matrix.shape[1])
              rag.index.add(matrix)
              save_faiss_index(rag.index)
            else:
              rag.index = None
              if os.path.exists('data/faiss.index'):
                os.remove('data/faiss.index')
                  
            st.success(f'{file_to_delete} berhasil dihapus')
            st.session_state.admin_page = 'main'
          except Exception as e:
            st.error(f'Gagal menghapus file: {e}')
            st.session_state.delete_file = None
            st.rerun()
        
      with confirm_col2:
        if st.button("Batal", key="confirm_delete_no"):
          st.session_state.confirm_delete = False
          st.session_state.file_to_delete = None
          st.session_state.admin_page = 'main'
          st.rerun()

    elif st.session_state.admin_page == 'data':
      st.subheader('📁 Data')
      with st.expander('Lihat File Indexed'):
        try:
          with open('indexed.json', 'r', encoding='utf-8') as f:
            indexed_files = json.load(f)
            if indexed_files:
              for name in indexed_files:
                st.markdown(f'{name}')
            else:
              st.info("Belum ada file yang diindex")
        except FileNotFoundError:
          st.info("Belum ada file yang diindex")  
        except Exception as e:
          st.error(f'Gagal memuat indexed file: {e}')
      with st.expander('Lihat Chunks:'):
        chunks_path = "data/chunks.json"
        if not os.path.exists(chunks_path):
          st.info("Belum ada chunks")
        else:
          try:
              with open("data/chunks.json", "r", encoding="utf-8") as f:
                chunks_dict = json.load(f)
                
                if not chunks_dict:
                  st.info("Belum ada chunks")
                else:
                  total_chunks = sum(len(chunk_list) for chunk_list in chunks_dict.values())
                  st.info(f'Total chunks: {total_chunks}')
                  for filename, chunks in chunks_dict.items():
                    st.markdown(f"### 📄 {filename}")
                    for i, chunk in enumerate(chunks, 1):
                      st.markdown(f"**Chunk {i}:**\n{chunk}\n")
                      st.markdown("---")
          except Exception as e:
            st.error(f"Gagal memuat chunks.json: {e}")
          
#--------------user interface
elif st.session_state.mode == "chat" and st.session_state.page == 'chat':
  st.header("Chatbot Layanan Akademik UG")
  st.text('Selamat datang, silahkan berikan pertanyaanmu dan chatbot akan membantumu menjawab pertanyaan terkait layanan akademik di Universitas Gunadarma😊.')

  query = st.text_input("Masukkan pertanyaan:")
  if query:
    rag = st.session_state.rag
    #retrieve relevant chunks
    if rag.index is None or not rag.chunks:
      st.warning("⚠️ Oops, ada masalah. Silahkan dicoba lagi nanti, ya! Untuk sementara, kamu bisa cek link ini : \n\n"
                 "- [BAAK](https://baak.gunadarma.ac.id/)\n"
                 "- [Studentsite](https://studentsite.gunadarma.ac.id/index.php/site/login)\n"
                 "- [Universitas Gunadarma](https://www.gunadarma.ac.id/)\n"
                 )
      st.stop()
    top_chunks = st.session_state.rag.retrieve_chunks(query)

    #build prompt
    prompt = create_prompt(query, top_chunks)

    #call OpenAI with streaming
    try:
      with st.spinner("UGbot sedang membuat jawaban..."):
        response = ask_openai(prompt)
      
        #streaming
        answer_container = st.empty()
        full_answer = ""
        for chunk in response:
          full_answer += chunk.choices[0].delta.content or ""
          answer_container.markdown(full_answer)
        
        #save to history
        st.session_state.chat_history.append({
          'question': query,
          'answer': full_answer,
        })
    except Exception as e:
      st.error(f'⚠️ Error: {e}')

#chat history menu
elif st.session_state.page == 'chat' and st.session_state.mode == 'riwayat chat':      
  st.header('🕘 Riwayat Chat')
  history = st.session_state.chat_history
  
  if history:
    for i, entry in enumerate(reversed(history[:]), 1):
      st.markdown(f"**Q{i} :** {entry['question']}")
      st.markdown(f"**A{i} :** {entry['answer']}")
      st.markdown("---")
  
    if st.button('Hapus Riwayat Chat'):
      st.session_state.chat_history = []
      st.success('Riwayat chat berhasil dihapus.')
  else:
    st.info('Belum ada riwayat chat.')
    
  