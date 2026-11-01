from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_output_tokens=1024,
    max_retries=3
)

result= chat.invoke("What is the capital of France?")

print(result.content[0]["text"])

print(result.usage_metadata["total_tokens"])


