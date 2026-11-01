from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1
)

response = llm.invoke("who built eiffel tower?")

print(response)