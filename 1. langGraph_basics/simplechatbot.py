from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def superbot(State: State):
    return {"messages":  [llm.invoke(State["messages"])]}

graph = StateGraph(State)
graph.add_node("superbot", superbot)

graph.add_edge(START, "superbot")
graph.add_edge("superbot", END)

graph_builder = graph.compile()

result = graph_builder.stream({"messages" : "Hello, I am Geet Desai. Will Arsenal wil today's match?"}, stream_mode="values")

for event in result:
    print(event)
