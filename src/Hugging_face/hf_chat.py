from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  
    task="text-generation",
    temperature=0.2,
    max_new_tokens=256, 
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("What is the capital of France?")

print(response.content)