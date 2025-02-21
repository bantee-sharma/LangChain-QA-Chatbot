from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Print all environment variables (for debugging)
print("Loaded Environment Variables:", os.environ)

# Fetch API token
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not api_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found! Make sure it's set in the .env file.")

print(f"Loaded API Token: {api_token[:5]}... (token partially hidden for security)")
