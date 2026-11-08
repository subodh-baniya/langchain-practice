from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=1.2,
    max_output_tokens=1024,
    max_retries=3
)

prompt_template = PromptTemplate(
    template="summarize the following poem : {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

def get_poem_text():
    text_loader = TextLoader("src/document_loader/hello.txt", encoding="utf-8")
    result = text_loader.load()
    return result[0].page_content

function_chain =  prompt_template | model | parser

result = function_chain.invoke({"text": get_poem_text()})

print(result)