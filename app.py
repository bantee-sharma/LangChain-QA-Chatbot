from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
import streamlit as st

# Load .env variables
load_dotenv()

# Fetch API token and check if it's loaded correctly
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found! Make sure it's set in the .env file.")

print(f"Loaded API Token: {api_token[:5]}... (token partially hidden for security)")

# Set token as an environment variable (optional)
os.environ['HUGGINGFACEHUB_ACCESS_TOKEN'] = api_token

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small"
,  # Open-sorce alternative
    task="text-generation",
    model_kwargs={"max_length": 128},
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)


model = ChatHuggingFace(llm=llm)

res = model.invoke("who is indian prime minister")

print(res.content)

