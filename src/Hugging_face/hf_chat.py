from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="ibm-granite/granite-4.2-30b",  
    task="text-generation",
    temperature=0.2,
    max_new_tokens=256, 

)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("What is the capital of France?")

print(response.content)