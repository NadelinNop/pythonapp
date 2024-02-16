from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader
import io
import os
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from docx import Document
import pandas as pd
import docx2txt

app = Flask(__name__)

account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
container_name = "northstar-fileupload"

connection_string = "DefaultEndpointsProtocol=https;AccountName=northstaraistorage;AccountKey=H6Wk9KVZfhc+KH2rAr2q5Cnz2l+Bkf90ZiGy/LyecXr3l33nrOtQTbhwueJYhjUbv31hTpq7UWja+AStXm5GOw==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

def extract_text_from_blob(blob_client):
    blob_data = blob_client.download_blob().readall()

    file_name = blob_client.blob_name
    if '.' in file_name:
        file_extension = file_name.split('.')[-1].lower()

        if file_extension == 'pdf':
            # Extract text from PDF
            pdf_stream = io.BytesIO(blob_data)
            pdf_reader = PdfReader(pdf_stream)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif file_extension == 'docx':
            try:
                doc = Document(io.BytesIO(blob_data))
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            except Exception as e:
                print(f"Error extracting text: {e}")
                return None
        elif file_extension == 'csv':
            # Extract text from CSV
            df = pd.read_csv(io.StringIO(blob_data.decode('utf-8')))
            text = df.to_string(index=False)
            return text
        elif file_extension == 'doc':
            # Extract text from DOC
            text = docx2txt.process(io.BytesIO(blob_data))
            return text


        else:
            # Handle other file types as needed
            return ""
    else:
        # Handle case when file extension is not found
        return ""

def process_text(texts):
    # Concatenate the list of texts into a single text
    combined_text = "\n".join(texts)

    # Split the combined text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(combined_text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

@app.route('/query', methods=['POST'])
def query():
    folder_path = request.form.get('folder_path')  # Assuming you pass the virtual folder path as a parameter
    prompt = request.form.get('prompt')

    if folder_path:
        try:
            texts = []

            # List blobs in the specified virtual folder (directory)
            blobs = container_client.list_blobs(name_starts_with=folder_path)

            for blob in blobs:
                # Process each blob and extract text
                blob_client = container_client.get_blob_client(blob.name)
                extracted_text = extract_text_from_blob(blob_client)
                texts.append(extracted_text)

            # Create the knowledge base object for all files in the folder
            knowledgeBase = process_text(texts)
            docs = knowledgeBase.similarity_search(prompt)

            llm = OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=2024)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=prompt)
                print(cost)

            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'No virtual folder path provided'})
@app.route('/check_env')
def check_env():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return f"OPENAI_API_KEY: {openai_api_key}"

@app.route('/', methods=['GET'])
def index():
    return 'Hello, this is the main main page!'
if __name__ == '__main__':
    app.run(debug=True)