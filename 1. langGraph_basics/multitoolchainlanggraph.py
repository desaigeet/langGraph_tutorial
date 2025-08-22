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

load_dotenv()

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
# print(arxiv.invoke("Attention is all you need"))

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
# print(wiki.invoke("Machine Learning"))

tools = [arxiv, wiki]

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
builder.add_edge("tools", END)
graph_builder = builder.compile()

initial_messages = [
    HumanMessage(content="What is machine learning?"),
    AIMessage(content="Machine learning is a field of artificial intelligence..."),
    HumanMessage(content="What is the research paper that pioneered transformers?"),
]


results = graph_builder.invoke(State(messages=initial_messages))
for msg in results["messages"]:
    print(msg.pretty_print())
