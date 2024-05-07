# Importing necessary modules required for the task
import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables securely
from dotenv import load_dotenv
load_dotenv()  # Assuming you have a .env file at the root of your project
os.environ["OPENAI_API_KEY"] = "api_key here"

# Initialize Flask application
app = Flask(__name__)

# Define a function to fetch and parse content from a given URL
def fetch_and_parse_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)
    except requests.RequestException as e:
        return str(e)
    
# Define a function to split the text into manageable chunks
def setup_retrieval_system(document):
    chunks = [document[i:i + 1000] for i in range(0, len(document), 1000)]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma.from_texts(chunks, embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Define a function to answer a query using the retriever
def answer_question(query, retriever):
    llm = OpenAI(model='babbage-002')
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = qa_chain({"query": query})
    return result

# Define a route to handle POST requests for question answering
@app.route('/fetch-answer', methods=['POST'])
def fetch_answer():
    try:
        data = request.get_json()
        document = fetch_and_parse_content(data['url'])
        if not document:
            return jsonify({"error": "Failed to fetch or parse content from the URL"}), 400
        retriever = setup_retrieval_system(document)
        answer = answer_question(data['question'], retriever)
        result_part = answer['result'].split("\n\nQuestion:")[0]
        return jsonify({"answer": result_part})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define a route to serve the main page
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)











