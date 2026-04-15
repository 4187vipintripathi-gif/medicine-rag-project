import os
from unittest import loader
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone,ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

# Load environment variables
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



#Creating index in pinecone
index_name = "my-index2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
else:
    print("Index already exists")

index = pc.Index("my-index2")


#Reading the pdf files and creating documents
Dir=r"C:\Users\4187v\Downloads\Medicine Project\Data"
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def extract_medicines(text):
    prompt = f"""
# You are a strict medical extractor.

# Task:
# i wil give the content of a medicine leaflet. Identify the only MAIN medicine this document is about.
# output only the one main medicine name as string.
# avoid any commentry around it.

Task:
Identify the PRIMARY medicine name this leaflet is about.

Rules:
- Prefer the medicine name mentioned in the TITLE or HEADER
- It is usually at the top of the document
- Ignore ingredients like Paracetamol, ethanol, etc.
- Return ONLY ONE name
- No explanation

Text:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    
    return raw

#Chunking the documents
def chunk(docs, chunk_size=500,chunk_overlap=100):
    text_splitters=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=text_splitters.split_documents(docs)
    return docs

def load_data(file_path):
    doc=read_pdf(file_path)
    medicine_name = extract_medicines(doc)
    documents=chunk(docs=doc)


#Creating records to be inserted in pinecone
    records = []

    for i, d in enumerate(documents):
            file_name = d.metadata["source"]
            file_name = os.path.basename(file_name)
            
            vector = embeddings.embed_query(d.page_content)
            records.append({
                "id": f"{file_name}_{i}",
                "values": vector,
                "metadata": {
                    "text": d.page_content,
                    "medicine_name": medicine_name
                }
            })

    data_size=100
    for i in range(0, len(records), data_size):
            data = records[i:i+data_size]
            
        
        
            index.upsert(
            namespace="example-namespace",
            vectors=data
        )


#load the data
def start_loading():
    for file in os.listdir(Dir):
        if file.endswith(".pdf"):
            file_path = os.path.join(Dir, file)
            load_data(file_path)

start_loading()