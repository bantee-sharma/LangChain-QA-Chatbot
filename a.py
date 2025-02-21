import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables (for local development)
load_dotenv()

# Get API token from Streamlit secrets (for deployment) or local environment
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or st.secrets.get("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    st.error("Hugging Face API token is missing. Please set it in Streamlit Secrets.")
    st.stop()

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

# Streamlit UI
st.title("🤖 LangChain Question Answering Model")
st.write("Ask me anything!")

question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    response = llm_chain.invoke({"question": question})
    st.write(response.get('text', 'No response received.'))
