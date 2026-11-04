from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model1=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

model2=ChatGroq(
    model="openai/gpt-oss-safeguard-20b",
    max_retries=1,
)


parser = StrOutputParser()

prompt_template1 = PromptTemplate(
    input_variables=["user_input"],
    template="""You have to generate a short note from the given text : {user_input}"""
)

prompt_template2 = PromptTemplate(
    input_variables=["user_input"],
    template="""You have to generate a quiz of 5 questions from the given text : {user_input}"""
)

prompt_template3 = PromptTemplate(
    input_variables=["note_generator", "quiz_generator"],
    template="""You have to merge the {note_generator} and  {quiz_generator} so that they appear one after another. """
)


parallel_chain = RunnableParallel(
    {
        "note_generator": prompt_template1 | model1| parser,
        "quiz_generator": prompt_template2 | model2 | parser
    }
)

last_chain = prompt_template3 | model1 | parser

merged_chain = parallel_chain | last_chain

result = merged_chain.invoke({"user_input": "Technology has become an essential part of modern life, changing how people communicate, learn, and work. Artificial intelligence, in particular, is transforming industries by automating repetitive tasks, analyzing complex information, and solving problems more efficiently. However, technology also brings challenges, such as privacy concerns, cybersecurity risks, and overdependence on digital tools. The key is to use technology responsibly while continuously developing the skills needed to adapt to an evolving world.'."})

print(result)


