import os
from dotenv import load_dotenv
import fitz
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

def extract_text(pdf_path):
    all_text = ""
    for file in pdf_path:
        with fitz.open(file) as pdf_doc:
            for page in pdf_doc:
                all_text += page.get_text("text")
    return all_text

def split_text_into_chunk(text):
    splitter=TokenTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=splitter.split_text(text)
    docs=[Document(page_content=chunk) for chunk in chunks]
    return docs

# embeddings=GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY)
# llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=GOOGLE_API_KEY)

def create_vectorstore(docs):
    # embeddings=GoogleGenerativeAIEmbeddings(
    #     model="models/embedding-001",google_api_key=GOOGLE_API_KEY
    # )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in docs] 
    vectorstore=FAISS.from_texts(texts,embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def create_pdf_qa_lcel_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers based on the provided context.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)
    chain = (
        {"context": retriever | RunnableLambda(lambda docs: "\n\n".join([d.page_content for d in docs])),
         "input": RunnablePassthrough()}
        | prompt
        | llm
        | RunnableLambda(lambda x: x.content)
    )
    return chain

def pdf_qa_interface(pdf_file,query,chat_history):
    text=extract_text(pdf_file)
    docs=split_text_into_chunk(text)
    retriever=create_vectorstore(docs)
    qa_chain=create_pdf_qa_lcel_chain(retriever)

    answer=qa_chain.invoke(query)
    chat_history.append((query,answer))
    return answer,chat_history


