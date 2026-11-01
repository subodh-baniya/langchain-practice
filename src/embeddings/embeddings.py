from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=32,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

result = embeddings.embed_query("What is the capital of France?")

print(result)
print("Dimensions:", len(result))