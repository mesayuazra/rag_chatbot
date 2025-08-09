import streamlit as st
import json
import hashlib
import os
from dotenv import load_dotenv

USER_DB = 'data/admin.json'
os.makedirs('data', exist_ok=True)

load_dotenv()
ADMIN_USERNAME = st.secrets['ADMIN_USERNAME']
ADMIN_PASSWORD = st.secrets['ADMIN_PASSWORD']

#hash the password
def hash_password(password):
  return hashlib.sha256(password.encode()).hexdigest()

#load admin
def load_admin():
  if os.path.exists(USER_DB):
    with open(USER_DB, 'r') as f:
      return json.load(f)
  return {}

#save admin
def save_admin(admin):
  with open(USER_DB, 'w') as f:
    json.dump(admin, f, indent=2)

def admin_register():
  st.subheader('Daftar Admin')
  username = st.text_input('Username')
  password = st.text_input('Password', type='password')
  confirm_password = st.text_input('Confirm password', type='password')
  
  if st.button('Daftar'):
    if not username or not password or not confirm_password:
      st.error('Isi semua kolom.')
      return
    
    if password != confirm_password:
      st.error('Password yang Anda masukkan salah.')
  
    admin = load_admin()
    if username in admin:
      st.error('Username sudah digunakan.')
      return
    
    admin[username] = hash_password(password)
    save_admin(admin)
    st.success('Akun berhasil dibuat. Silahkan login.')

#login admin
def login():
  st.subheader("Admin Login")
  username = st.text_input('Username')
  password = st.text_input('Password', type='password')
  
  if st.button('Login'):
    # admin = load_admin()
    # hashed_password = hash_password(password)
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
      st.session_state.logged_in = True
      st.success(f'Selamat datang, {username}')
    else:
      st.error('Username atau password salah.')