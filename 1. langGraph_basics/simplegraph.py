from typing_extensions import TypedDict
import random
from typing import Literal
from langgraph.graph import StateGraph, START, END

class state(TypedDict):
    graph_info: str

def start_play(state: state):
    print("start_play has been called")
    return {"graph_info": state["graph_info"]}

def play_cricket(state: state):
    print("play_cricklet has been called")
    return {"graph_info": state["graph_info"] + "I am playing cricket"}

def play_badminton(state: state):
    print("play_badminton has been called")
    return {"graph_info": state["graph_info"] + "I am playing badminton"}

def decide_game(state: state) -> Literal["play_cricket", "play_badminton"]:
    print("decide_game has been called")

    graph_info = state["graph_info"]
    if random.random() > 0.5:
        return "play_cricket"
    else:
        return "play_badminton"
    
graph = StateGraph(state)
graph.add_node("start_play", start_play)
graph.add_node("play_cricket", play_cricket)
graph.add_node("play_badminton", play_badminton)

graph.add_edge(START,"start_play")
graph.add_conditional_edges("start_play",decide_game)
graph.add_edge("play_cricket", END)
graph.add_edge("play_badminton", END)

graph_builder = graph.compile()

result = graph_builder.invoke({"graph_info": "I am planning to play "})
print(result)