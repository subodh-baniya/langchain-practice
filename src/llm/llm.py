from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1
)

response = llm.invoke("Write a short story about a robot learning to fight crime in a futuristic city.")

print(response)