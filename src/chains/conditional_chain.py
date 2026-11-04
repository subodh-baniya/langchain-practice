from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

class Review(BaseModel):

    semantic: Literal["positive", "negative"]=Field(description="semantic of the product review, either positive or negative")

   
review_parser = PydanticOutputParser(pydantic_object=Review)
parser = StrOutputParser()

prompt_template1 = PromptTemplate(
    input_variables=["user_input"],
    template="""Based on the following product review, determine whether the semantic of the review..
Product Review: {user_input} \n {format_instructions}""",
    partial_variables={"format_instructions": review_parser.get_format_instructions()}
    )

prompt_template2 = PromptTemplate(
    input_variables=["user_input"],
    template="""If input is positive, generate a thankful message for the product review. Product Review: {user_input}""")

prompt_template3 = PromptTemplate(
    input_variables=["user_input"],
    template="""If input is negative, generate a message to the user apologizing for the product review. Product Review: {user_input}""")

chain1 = prompt_template1 | model | review_parser

review_chain = RunnableBranch(
    (lambda x: x.semantic == "positive", prompt_template2 | model | parser),
    (lambda x: x.semantic == "negative", prompt_template3 | model | parser),    
    RunnableLambda(lambda x: "The semantic of the product review is not positive or negative."))

merged_chain = chain1 | review_chain

result = merged_chain.invoke({"user_input": "The Sony WH-1000XM5 delivers an impressive combination of premium sound quality, adaptive noise cancellation, and ergonomic design. Its ability to isolate ambient noise makes it particularly effective in crowded environments, while the balanced audio profile provides a rich listening experience across different genres. However, the non-foldable design compromises portability, and the premium price may not justify the upgrade for users who already own high-end headphones. Overall, it is a well-engineered product that excels in performance but has a few practical limitations."})

print(result)



