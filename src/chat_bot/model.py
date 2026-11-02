from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

models = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)