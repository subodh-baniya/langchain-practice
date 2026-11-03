from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

pipeline = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

model = ChatHuggingFace(llm=pipeline)

parser = JsonOutputParser()

template = PromptTemplate(
    input_variables=[],
    template="""give me an mock user info including name age email:{user_input}""",
    partial_variables={"user_input": parser.get_format_instructions()},

)

prompt = template.format()

result=model.invoke(prompt)

print(result)

result_json=parser.parse(result.content)

print(result_json)