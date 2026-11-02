from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.7,
)

template = PromptTemplate(
    input_variables=["user_input"],
    template="You are a helpful assistant and always give answer on to the point exact and concise not exceeding 3 short sentences. Answer the following question: {user_input}"
)

chain=template|model

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    response = chain.invoke({"user_input": user_input})
    print(f"Bot: {response.content[0]['text']}")

