from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import random
from dataclasses import dataclass
from pydantic import BaseModel

class State(BaseModel):
    name: str
    game: Literal["cricket","badminton"]

def play_game(state: State):
    print("-- Playing Game --")
    return State(name=state.name + " wants to play", game=state.game)

def cricket(state: State):
    print("--playing cricket--")
    return State(name=state.name, game="cricket")

def badminton(state: State):
    print("-- playing badminton --")
    return State(name=state.name, game="badminton")

def decide_game(state: State) -> Literal["cricket", "badminton"]:
    print("-- Deciding the game --")
    if random.random() > 0.5:
        return "cricket"
    else:
        return "badminton"

builder = StateGraph(State)
builder.add_node("play_game", play_game)
builder.add_node("cricket", cricket)
builder.add_node("badminton", badminton)

builder.add_edge(START, "play_game")
builder.add_conditional_edges("play_game", decide_game)
builder.add_edge("cricket", END)
builder.add_edge("badminton", END)

graph_builder = builder.compile()
result = graph_builder.invoke(State(name="Geet Desai", game="cricket"))
print(result)