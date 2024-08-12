import streamlit as st
import os
import joblib

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from langchain_openai import OpenAIEmbeddings
from IPython.display import Markdown, display
import chromadb

from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from cryptography.fernet import Fernet

# Set API keys
llama_cloud_api = "llx-PUHGYTgvb26eAsGtyRevZmk6iYNt0Qs49XuNrgVdSwhW1scL"
cohere_api = "2dChAVyhW2jYng7tWYGuh5ol0xiKqDTVKtBT4A2W"
Secret_Key_openai = "X1A22EgQilQO_-w4jOTflmpG8o8BXMtZYTcsJa3yoY8="
Encrypted_openai = "gAAAAABmuJ6MrTyGsbVCg_Y7zkuH5VAG75QTPfW7lzNPaR8XpoJEEVhgwO35THTQ9HYpSTzVI0eWBybfg5yqKiPVKx_1I1TmGGo28ztJycn5cih8HU7OAOyl1yRHd4dSZK1AT79l6t5B531-xYoi2j8wWy83MCO6nNqofe0hZkcoTVaBpmt7mZy-7NuURk1-5bMkEKbvoUESFqI3MYkj_WEH73AXHN2c3S3sDBzak2HaAlsEcUN5U59SmSAsFXD5v_lfqK79iXS4"

# Decrypt API key
def demask_api_key(encrypted_key, key):
    fernet = Fernet(key)
    decrypted_key = fernet.decrypt(encrypted_key).decode()
    return decrypted_key

demasked_key = demask_api_key(Encrypted_openai, Secret_Key_openai)

os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_api
os.environ["OPENAI_API_KEY"] = demasked_key
os.environ["COHERE_API_KEY"] = cohere_api

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Function definitions...
# (Include the rest of your functions here, including load_or_parse_data, create_vector_database_updated, etc.)

def main_updated():
    # Set the page configuration
    st.set_page_config(page_title="MOMRA Retrieve Smart AI", page_icon=":mag:", layout="wide")

    # Initialize session state for file storage
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    # Add a title with custom styling
    st.markdown(
        """
        <div style="text-align:center; padding: 20px;">
            <h1 style="color:#4CAF50; font-size:48px;">MOMRA Retrieve Smart AI</h1>
            <p style="font-size:24px;">Upload your PDF and ask anything!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # File upload
    st.sidebar.title("Upload Section")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    # Store the uploaded file in session state
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # Text input
    st.sidebar.title("Query Section")
    user_query = st.sidebar.text_input("What details can I pull out for you?")

    if st.sidebar.button("Submit"):
        if st.session_state.uploaded_file is not None and user_query:
            # Read the uploaded PDF file
            file_name = str(st.session_state.uploaded_file.name)
            pdf_bytes = st.session_state.uploaded_file.read()

            # Display the file name and query for clarity
            st.sidebar.success(f"File uploaded: {file_name}")
            st.sidebar.success(f"Query: {user_query}")

            # Pass the file and query to the function
            result = overall_main_func(pdf_bytes, user_query, file_name)
            
            # Display the result as markdown with custom styling
            st.markdown(
                """
                <div style="text-align:center; padding: 20px;">
                    <h2 style="color:#FF5722;">Result</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style='padding: 20px; background-color:#f9f9f9; border-radius:10px;'>
                    <p style='font-family:Arial, sans-serif; font-size:20px; font-weight:bold; color:#333;'>{result}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.error("Please upload a PDF file and enter a query.")

    # Add a footer
    st.markdown(
        """
        <div style="text-align:center; padding: 20px;">
            <hr>
            <p style="font-size:16px;">Powered by MOMRA Retrieve Smart AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main_updated()
