from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from pydantic import BaseModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
# print(arxiv.invoke("Attention is all you need"))

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
# print(wiki.invoke("Machine Learning"))

def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers a and b
    """
    return a*b

def divide(a: int, b: int) -> int:
    """
    Divide two number a and b
    """
    return a/b

tools = [arxiv, wiki, multiply, divide]

llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)

class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]

def tool_calling_llm(state: State):
    response = llm_with_tools.invoke(state.messages)
    return {"messages" : state.messages + [response]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)

builder.add_edge("tools","tool_calling_llm")
# builder.add_edge("tools", END)

# we enable memory saver be adding a checkpointer while compiling graph

memory = MemorySaver()
graph_builder = builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"1"}}

initial_messages = [
    HumanMessage(content="Multiply 5 with 5."),
]


results = graph_builder.invoke(State(messages=initial_messages), config=config)
for msg in results["messages"]:
    print(msg.pretty_print())

next_messages = [
    HumanMessage(content="Divide that number by 25."),
]
results = graph_builder.invoke({"messages" : next_messages}, config=config)

for msg in results["messages"]:
    print(msg.pretty_print())
