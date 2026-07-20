import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")
print(f"Loaded key: {api_key[:5]}...{api_key[-5:]} (Length: {len(api_key)})")

genai.configure(api_key=api_key)

try:
    print("Testing gemini-2.5-flash...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say 'Hello world' and nothing else.")
    print("Success! Response:", response.text)
except Exception as e:
    print("Error with gemini-2.5-flash:", type(e).__name__, str(e))
