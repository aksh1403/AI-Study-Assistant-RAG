from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
