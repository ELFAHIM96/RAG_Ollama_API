# import langchain llm ollama
import os
from flask import Flask, request, jsonify

from langchain_community.llms import  Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


app = Flask(__name__)


folder_path = "db"
cached_llm = Ollama(model="llama3")


embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =1024, chunk_overlap =80, length_function = len, is_separator_regex= False)

raw_prompt = PromptTemplate.from_template("""
<s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST]</s>
[INST] {input}      
         Context: {context}
        Answer:
 [/INST]
                                          """)
#text_splitter = RecursiveCharacterTextSplitter(max_length=1000)

#respance =llm.invoke("tell me a cat joke")

#print(respanse)
@app.route("/")
def home():
    return "Welcome to the AI API! Use the /ai endpoint to interact with the model."


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called ")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response =cached_llm.invoke(query)

    response_answer = {"answer": response}
    print(response)
    return response_answer

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called ")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print('loading vector store')

    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    print("Creating Chain")
    retriever= vector_store.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {
            "k":20,
            "score_threshold":0.5,
        },
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input":query})
    response_answer = {"answer":result}
    return response_answer

@app.route("/pdf", methods=["POST"])
def pdfPost():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    file_name = file.filename
    save_file = "pdf/", file_name
    file.save(save_file)
    loader = PDFMinerLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")


    vector_store = Chroma.from_documents(
        documents= chunks, embedding= embedding, persist_directory= folder_path
    )
    vector_store.persist()


    response = {"status": "Successfully Uploaded", 
                "filename": file_name, 
                "doc_len":len(docs), 
                "chunks": len(chunks)}
    return response



def start_app():
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    start_app()