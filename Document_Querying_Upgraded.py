import streamlit as st
import os
import tempfile
# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from langchain_openai import OpenAIEmbeddings
from IPython.display import Markdown, display
import chromadb

from llama_parse import LlamaParse
import langchain_core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#
#from groq import Groq
#from langchain_groq import ChatGroq
#
import joblib
import os
import nest_asyncio  # noqa: E402
nest_asyncio.apply()

from io import BytesIO
import pickle
import re
from pydantic import BaseModel
from typing import List
import os
from langchain_community.document_loaders import PyPDFLoader
import openai
from langchain.llms import OpenAI
import gensim
import nltk
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from pypdf import PdfReader
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
import json
import math

Secret_Key_openai="1EapbgJS7W8jPQjNEWkFbOKg9DOq0koS2TaRnUNkoTk="
Encrypted_openai="gAAAAABmu2X_UzAX1-ZfJLRl__U73qWwov2l2fWka-ux3dcZ-a8LbVhi2Yg1mTOh4NPMtGRiKssw4aEElPm3gEGFDPfPMoJQf4MqflEAgTlsrI4q26d2FxCWc-Sb0AwUeEhOX_8JbDqAx5IrB-UDCuxLVfQuEFDycfWcr8QRLcOpaNWy3XmvbZzC0dvXYLhXkWTEt9KAewhL2MktzyPXyDqUItnqOdI2rO3DYLI02z7y9NnHe0I_09ep5RbxEajZu9Q_HkyosZzS"

def demask_api_key(encrypted_key, key):
    fernet = Fernet(key)
    decrypted_key = fernet.decrypt(encrypted_key).decode()
    return decrypted_key

demasked_key = demask_api_key(Encrypted_openai, Secret_Key_openai)


llama_cloud_api="llx-PUHGYTgvb26eAsGtyRevZmk6iYNt0Qs49XuNrgVdSwhW1scL"
cohere_api="2dChAVyhW2jYng7tWYGuh5ol0xiKqDTVKtBT4A2W"
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader

import os
# API access to llama-cloud
os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_api

# Using OpenAI API for embeddings/llms
os.environ["OPENAI_API_KEY"] = demasked_key

# Using Cohere for reranking
os.environ["COHERE_API_KEY"] = cohere_api

embeddings = OpenAIEmbeddings()


llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


def preprocess(text, stop_words):
    result = []
    for token in simple_preprocess(text, deacc=True):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
    
#

def clean_lines(lines):
    cleaned = []
    for line in lines:
        cleaned = []
        for line in lines:
            # Remove leading numbers and periods using regex
            cleaned_line = re.sub(r'^\d+\.\s*-|-', '', line).strip()
            if cleaned_line:  # Only add non-empty lines
                cleaned.append(cleaned_line)
        return cleaned

def topics_chain(llm, file, num_topics, words_per_topic):
    list_of_topicwords = get_topic_lists_from_pdf(file, num_topics, words_per_topic)
    print(len(list_of_topicwords))
    print("list of topic words")
    print(list_of_topicwords)
    string_lda = ""
    for list in list_of_topicwords:
        string_lda += str(list) + "\n"

    template_string = '''You are a language model that identifies common topic in lists of words. For each list provided, analyze the words and return one representative topic that encapsulates the main theme of that list. 

    Please provide five to ten topics for the list that represents the common theme. Return as a List without number indexes. 

       

        Lists: """{string_lda}""" '''

    print(template_string)
    prompt_template = ChatPromptTemplate.from_template(template_string)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"string_lda": string_lda, "num_topics": num_topics})
    print(response)
    print(type(response))
    return response

def get_topic_lists_from_pdf(file_path, num_topics=7, words_per_topic=11):
    loader = PdfReader(file_path)
    documents = []
    for page in loader.pages:
        documents.append(page.extract_text())

    stop_words = set(stopwords.words(['english', 'spanish']))
    processed_documents = [preprocess(doc, stop_words) for doc in documents]

    dictionary = corpora.Dictionary(processed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in processed_documents]

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=words_per_topic)

    topics_ls = []
    for topic in topics:
        words = topic[1].split("+")
        topic_words = [word.split("*")[1].replace('"', '').strip() for word in words]
        topics_ls.append(topic_words)

    return topics_ls 
    

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def overall_main_func(pdffile, user_query,file_name):
    print("Entering Overall Main Function--")
    vs, embed_model = create_vector_database_updated(pdffile,file_name)
    
    retriever = vs.as_retriever(search_kwargs={'k': 3})
    prompt = set_custom_prompt()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    response = qa.invoke({"query": user_query})
    return response.get('result')

def load_or_parse_data(file_pdf,file_name):
    
    print("Loading or Parsing Data--")
    
    data_file=file_name+".pkl"
    

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        print("LLama Parsing")
        parsingInstructionBrd = """The provided document may be from any domain. It may be a BRD, Process Document, Inspection Document, Technical document etc.,
        This form provides detailed user stories about the current project.
        Parse the Documents and 
        Try to be precise while answering the questions. Do not generate answers if you cannot find the answer from the document."""
        parser = LlamaParse(api_key=llama_cloud_api,
                            result_type="markdown",
                            parsing_instruction=parsingInstructionBrd,
                            max_timeout=5000,)
                         
        llama_parse_documents = parser.load_data(file_name)


        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data

def create_vector_database_updated(file_pdf, file_name):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    print("Create Vector Database")
    persist_dir_name = file_name + "_chroma_db_llamaparse"

    # Check if the Chroma database already exists
    if os.path.exists(persist_dir_name):
        print(f"Chroma database already exists at {persist_dir_name}. Loading existing database.")
        #vs = Chroma(persist_directory=persist_dir_name, collection_name="rag")
        vs = Chroma(persist_directory=persist_dir_name, embedding_function=embeddings,collection_name="rag")
        #embedding = embeddings  # Assign your embeddings model if needed
    else:
        # Call the function to either load or parse the data
        print("new pdf parsing--")
        llama_parse_documents = load_or_parse_data(file_pdf, file_name)
        #print(llama_parse_documents[0].text[:300])
        markdown_name = file_name.replace(".pdf", "") + ".md"
        print("markdownname")
        print(markdown_name)
        with open(markdown_name, 'a', encoding='utf-8') as f:  # Open the file in append mode ('a')
            for doc in llama_parse_documents:
                f.write(doc.text + '\n')

        loader = UnstructuredMarkdownLoader(markdown_name)
        documents = loader.load()
        # Split loaded documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        print(f"length of documents loaded: {len(documents)}")
        print(f"total number of document chunks generated :{len(docs)}")

        # Initialize Embeddings
        # embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        # Create and persist a Chroma vector database from the chunked documents
        print(persist_dir_name)
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir_name,
            collection_name="rag",
        )

        # Persist the Chroma database
        vs.persist()

    # Return the vector store
    return vs


def run_pdf_document_query(file, question, num_topics=10, words_per_topic=11):

    vectorstore = create_vector_database_updated(file, file_name=file)

    retriever = vectorstore.as_retriever()
    custom_prompt_template = set_custom_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt_template},
    )

    result = qa_chain({"query": question})
    answer = result["result"]
    source = result["source_documents"]

    cleaned_source = []
    for doc in source:
        if isinstance(doc, Document):
            cleaned_lines = clean_lines(doc.page_content.splitlines())
            cleaned_source.append("\n".join(cleaned_lines))

    # Convert the list of strings into a string
    cleaned_source = "\n\n".join(cleaned_source)

    return answer
    # Topics Chain
    # topics = topics_chain(llm, file, num_topics, words_per_topic)
    # return answer, cleaned_source, topics

def get_total_pages(pdf_file):
    reader = PdfReader(pdf_file)
    total_pages = len(reader.pages)
    return total_pages
    
def get_topics_function(uploaded_file, num_topics=7, words_per_topic=11):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            pdffilepath = tmp_file.name
        summary = topics_chain(llm, pdffilepath, num_topics, words_per_topic)
        print("summary----->")
        print(summary)
        return summary.splitlines()

def get_summary_function(uploaded_file):
    prompt_template = """
    Summarize all the pages of following Document within a limit of 250 tokens:

    {text}

    Summary:
    """
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            pdffilepath = tmp_file.name
            loader = PyPDFLoader(pdffilepath)
            documents = loader.load()
            total_pages = len(documents)
            print("total pages in document")
            print(total_pages)
            pages_to_extract = math.ceil(total_pages * 0.3) 
            print("pages to extract")
            print(pages_to_extract)
            summary_doc = documents[:pages_to_extract]
             # Extract text from the filtered pages
            pdf_text = [doc.page_content for doc in summary_doc]

            # Combine the text into a single string
            combined_text = "\n".join(pdf_text)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            
            chunks = text_splitter.split_text(combined_text)
            
            #chain = load_summarize_chain(llm, chain_type="map_reduce",prompt=PromptTemplate(template=prompt_template, input_variables=["text"]))
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            result = chain.invoke(summary_doc)
            print(result["output_text"])
            
            return result["output_text"]



# Define the streamlit interface

def main_updated():
    # Set the page configuration
    st.set_page_config(page_title="MOMRA Retrieve Smart AI", page_icon=":mag:", layout="wide")

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

    # Initialize session state for uploaded file and result
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'topics' not in st.session_state:
        st.session_state.topics = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None

    # File upload
    st.sidebar.title("Upload Section")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
    
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
            
            # Save the uploaded file to the current working directory
            file_path = os.path.join(os.getcwd(), file_name)
        
            # Write the uploaded file to the specified path
            with open(file_path, "wb") as f:
                f.write(pdf_bytes)
        
            # Store the file path in session state
            st.session_state.uploaded_file_path = file_path

            # Display the file name and query for clarity
            st.sidebar.success(f"File uploaded: {file_name}")
            st.sidebar.success(f"Query: {user_query}")

            # Pass the file and query to the function
            st.session_state.result = run_pdf_document_query(file_name, user_query)
            
    # Display the result if available
    if st.session_state.result is not None:
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
                <p style='font-family:Arial, sans-serif; font-size:20px; font-weight:bold; color:#333;'>{st.session_state.result}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # New Button for Extracting Document Topics
    if st.sidebar.button("Extract Document Topics"):
        if st.session_state.uploaded_file is not None:
            st.session_state.topics = get_topics_function(st.session_state.uploaded_file)
            st.markdown("**Extracted Topics:**")
            for topic in st.session_state.topics:
                st.markdown(f"- **{topic}**")
        else:
            st.error("Please upload a PDF file to extract topics.")
            
    # New Button for Generating Document Summary
    if st.sidebar.button("Generate Document Summary"):
        if st.session_state.uploaded_file is not None:
            st.session_state.summary = get_summary_function(st.session_state.uploaded_file)
            st.markdown("**Document Summary:**")
            st.write(st.session_state.summary)
        else:
            st.error("Please upload a PDF file to generate a summary.")

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
