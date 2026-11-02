from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st  

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_output_tokens=1024,
    max_retries=3
)

st.title("LLM Streamlit App")    

question=st.text_input("Enter your question: ")

template=PromptTemplate(
    template="You are a helpful assistant. Answer the following question: {question}",
    input_variables=["question"]
)

prompt=template.invoke({"question": question})

result= model.invoke(prompt)

st.write(result.content[0]["text"])






