from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("Starting Gemini test...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    timeout=30,
    max_retries=0
)

print("Calling Gemini...")

response = llm.invoke("What is machine learning?")

print("Response received!")
print(response.content)