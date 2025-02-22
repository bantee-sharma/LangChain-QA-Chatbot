from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

app = Flask(__name__)

# Load Hugging Face Model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    model_kwargs={"max_length": 128},
    temperature=0.7,
    HUGGINGFACEHUB_ACCESS_TOKEN=hf_token
)

# Define Prompt Template
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "LangChain Flask API is running!"})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    response = llm_chain.invoke({"question": question})
    return jsonify({"answer": response.get("text", "No response received.")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
