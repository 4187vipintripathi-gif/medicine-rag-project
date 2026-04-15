from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
import fitz  # PyMuPDF
from transformers import pipeline

from dotenv import load_dotenv
load_dotenv()
Dir=r"C:\Users\4187v\Downloads\Medicine Project\Data"
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("my-index2")
appli = Flask(__name__)
CORS(appli)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrive_result(query,k=4):
    vectores=embeddings.embed_query(query)
    matching_result=index.query(vector=vectores,top_k=k, namespace="example-namespace", include_metadata=True)
    return matching_result

def retrieve_filtered_chunks(query, medicine, k=4):
    vectores=embeddings.embed_query(query)
    matching_result=index.query(vector=vectores,top_k=k, namespace="example-namespace", include_metadata=True, filter={"medicine_name": medicine})
    return matching_result

def answer_user_question(query, medicine, improvised_query):
    client = Groq(api_key=groq_api_key)
    results = retrieve_filtered_chunks(improvised_query, medicine, k=4)

    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer only from the given context. do not use any information that is not in the context. If you don't know the answer, say you don't know."
        },
        {
            "role": "user",
            "content": f"""
        Context:
        {context}

        Question:
        {query}
        """
        }
    ],
    temperature=0
)
    return response.choices[0].message.content

def get_unique_medicines(query,k=83):
    results = retrive_result(query,k)

    Dawai_naam =set()

    for med in results["matches"]:
        metadata = med["metadata"]
        if "medicine_name" in metadata:
            Dawai_naam.add(metadata["medicine_name"])
    
    # {
    #     match.get("metadata", {}).get("medicine_name")
    #     for match in results.get("matches", [])
    #     if match.get("metadata", {}).get("medicine_name")
    # }

    return Dawai_naam

def improvise_user_question(query):
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Improvise the user's question to make it more clear and concise. do not add any commentry. do not change the meaning of the question. only improvise the question to make it more clear and concise. retrun five improvised questions saparated by new line. do not add any commentry."
        },
        {
            "role": "user",
            "content": f"""
        Question:
        {query}
        """
                }
            ],
            temperature=0
        )
    return response.choices[0].message.content

@appli.route('/', methods=['GET'])
def welcome():
    return render_template('medichat.html')

@appli.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    print("Received data:", data)
    
    query = data.get("question","")
    medicine = data.get("medicine")

    print("Original Question:", query)
    print("Selected Medicine:", medicine)
    improvised_query = improvise_user_question(query)
    print("Improvised Question:", improvised_query)

    improvised_query = query + " " + medicine
    answer = answer_user_question(query, medicine, improvised_query)

    return jsonify({
        "answer": answer
    })


ALL_MEDICINES = list(get_unique_medicines("all medicines name"))
@appli.route('/medicines', methods=['GET'])
def medicines():
    return jsonify({
        "medicines": ALL_MEDICINES
    })

if __name__ == '__main__':
    appli.run(host="0.0.0.0", port=7860)