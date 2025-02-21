import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()
os.environ['HUGGINGFACEHUB_ACCESS_TOKEN'] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Load Hugging Face Model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    model_kwargs={"max_length": 128},  # Fixed max_length issue
    temperature=0.7,
    HUGGINGFACEHUB_ACCESS_TOKEN=os.environ['HUGGINGFACEHUB_ACCESS_TOKEN']
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
    st.write(response['text'])  # Corrected response extraction
