import streamlit as st
import os
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


open_ai_api="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"
llama_cloud_api="llx-PUHGYTgvb26eAsGtyRevZmk6iYNt0Qs49XuNrgVdSwhW1scL"
cohere_api="2dChAVyhW2jYng7tWYGuh5ol0xiKqDTVKtBT4A2W"
from langchain.chat_models import ChatOpenAI

import os
# API access to llama-cloud
os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_api

# Using OpenAI API for embeddings/llms
os.environ["OPENAI_API_KEY"] = open_ai_api

# Using Cohere for reranking
os.environ["COHERE_API_KEY"] = cohere_api

embeddings = OpenAIEmbeddings()


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
#

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def load_or_parse_data(file_pdf,file_name):
    
    data_file=file_name+".pkl"
    

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstructionBrd = """The provided document is a Business Requirement Document prepared by Business Analyst.
        This form provides detailed user stories about the current project.
        It includes document purpose, scope of work, assumptions, business processes, pre-requisites and user stories.
        It contains many tables.
        Try to be precise while answering the questions"""
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
    persist_dir_name = file_name + "_chroma_db_llamaparse"

    # Check if the Chroma database already exists
    if os.path.exists(persist_dir_name):
        print(f"Chroma database already exists at {persist_dir_name}. Loading existing database.")
        #vs = Chroma(persist_directory=persist_dir_name, collection_name="rag")
        vs = Chroma(persist_directory=persist_dir_name, embedding_function=embeddings,collection_name="rag")
        #embedding = embeddings  # Assign your embeddings model if needed
    else:
        # Call the function to either load or parse the data
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
            persist_directory=persist_dir_name,  # Local mode with in-memory storage only
            collection_name="rag"
        )

        print('Vector DB created successfully!')

    return vs, embeddings
def create_vector_database(file_pdf,file_name):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data(file_pdf,file_name)
    print(llama_parse_documents[0].text[:300])
    markdown_name=file_name.replace(".pdf","")+".md"
    print("markdownname")
    print(markdown_name)
    with open(markdown_name, 'a',encoding='utf-8') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    #output_md_name="output_"+markdown_name
    #markdown_path = "output.md"
    loader = UnstructuredMarkdownLoader(markdown_name)

   #loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    #len(docs)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")
    #docs[0]

    # Initialize Embeddings
    #embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create and persist a Chroma vector database from the chunked documents
    persist_dir_name=file_name+"_chroma_db_llamaparse"
    print(persist_dir_name)
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir_name,  # Local mode with in-memory storage only
        collection_name="rag"
    )

    #query it
    #query = "what is the agend of Financial Statements for 2022 ?"
    #found_doc = qdrant.similarity_search(query, k=3)
    #print(found_doc[0][:100])
    #print(qdrant.get())

    print('Vector DB created successfully !')
    return vs,embeddings


def overall_main_func(pdffile, user_query,file_name):
    
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

def main():
    st.title("MOMRA Retrieve Smart AI")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    # Text input
    user_query = st.text_input("What details can I pull out for you?:")
    
    if st.button("Submit"):
        if uploaded_file is not None and user_query:
            # Read the uploaded PDF file
            file_name = str(uploaded_file.name)
            #new_filename = file_name.replace(".pdf", "")
            #print(new_filename)  # Output: temp
            pdf_bytes = uploaded_file.read()
	        

            # Pass the file and query to the function
            result = overall_main_func(pdf_bytes, user_query,file_name)
            
            # Display the result as markdown
            st.markdown("### Result")
            st.markdown(result)
        else:
            st.error("Please upload a PDF file and enter a query.")
            
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

    # File upload
    st.sidebar.title("Upload Section")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
    
    # Text input
    st.sidebar.title("Query Section")
    user_query = st.sidebar.text_input("What details can I pull out for you?")

    if st.sidebar.button("Submit"):
        if uploaded_file is not None and user_query:
            # Read the uploaded PDF file
            file_name = str(uploaded_file.name)
            pdf_bytes = uploaded_file.read()

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
