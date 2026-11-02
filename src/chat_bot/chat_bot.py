from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

template = PromptTemplate(
    input_variables=["user_input"],
    template="You are a helpful assistant and always give answer on to the point exact and concise not exceeding 3 short sentences. Answer the following question: {user_input}"
)

chain=template|model

chathistory = []

while True:
    user_input = input("You: ")
    chathistory.append(user_input)
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    response = chain.invoke(chathistory)
    chathistory.append(response.content[0]["text"])
    print(f"Bot: {response.content[0]['text']}")

