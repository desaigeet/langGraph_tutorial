from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from pydantic import BaseModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo")

@tool
def add_numbers(x: int, y: int) -> int:
    """
    Performes addition of two numbers and return the result.
    """
    print("Tool called with x:", x, "and y:", y)
    return x + y
llm_with_tool = llm.bind_tools([add_numbers])

class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]


initial_messages = [HumanMessage(content="Hello, how are you?")]
initial_messages.append(AIMessage(content="I'm fine, thank you!"))
initial_messages.append(HumanMessage(content="What is machine learning?"))

def llm_tool(state: State) -> State:
    response = llm_with_tool.invoke(state.messages)
    return {"messages": state.messages + [response]}

builder = StateGraph(State)
builder.add_node("llm_tool", llm_tool)
builder.add_node("tools", ToolNode([add_numbers]))

builder.add_edge(START, "llm_tool")
builder.add_conditional_edges(
    "llm_tool", 
    tools_condition
    )
builder.add_edge("tools", END)

graph_builder = builder.compile()

result = graph_builder.invoke(State(messages=initial_messages))
print(result)
