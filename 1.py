import toml

# Manually load secrets.toml to check if it's valid
with open(".streamlit/secrets.toml", "r") as f:
    secrets_data = toml.load(f)

print(secrets_data)  # This should print the token

