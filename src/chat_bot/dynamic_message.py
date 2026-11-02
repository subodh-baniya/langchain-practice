from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from .model import models

prompt=ChatPromptTemplate([
 ("system","You are an customer support assistant."),
 MessagesPlaceholder(variable_name="chat_history"),
 ("human","{user_input}"),
])

chat=[]

# Add messages to the chat history from the previous conversation
with open("src/chat_bot/chat.txt", "r") as file:
    for line in file:
        chat.append(line.strip())


template=prompt.invoke({"chat_history": chat, "user_input": "what is my order id?"})

result=models.invoke(template)

print(result.content[0]["text"])
