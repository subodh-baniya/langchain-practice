from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

pipeline = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

model = ChatHuggingFace(llm=pipeline)

template1 = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an expert research analyst with strong knowledge of science,
technology, society, and global issues.

Write a clear, informative, and factually accurate report on the following topic:

Topic: {user_input}

Instructions:
- Write approximately 10 sentences.
- Explain the topic and its underlying causes.
- Discuss its major effects and implications.
- Include relevant facts and examples where appropriate.
- Maintain a logical flow between sentences.
- Use clear, professional, and easy-to-understand language.
- Avoid unnecessary repetition and vague statements.
- Do not include an introduction or conclusion heading.

Output only the report.
"""
)

template2 = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an expert information analyst.

Analyze the following research report and extract exactly
two key points that communicate its most important findings.

Research Report:
{user_input}

Instructions:
- Each point must capture a distinct and important idea.
- Prioritize the most significant findings and implications.
- Write each point as one concise but informative sentence.
- Preserve the original meaning and factual accuracy.
- Avoid repeating information.
- Do not introduce information that is absent from the report.

Output format:

1. [First key point]
2. [Second key point]

Output only the two points.
"""
)
parser = StrOutputParser()

chain = template1 | model|parser |template2 | model|parser

result = chain.invoke({"user_input": "Global warming and its impact on the environment"})

print(result)



