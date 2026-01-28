import os
import sys


from dotenv import load_dotenv
from utilities.DataManager import extract_data
from utilities.AgentsGroup import load_llm
from models.AgentsGraph import AgentsGraph
from models.AgentState import AgentState

load_dotenv()


#Model configs
credentials = os.getenv("API_KEY")
model_id = "gpt-4o-mini"
end_p = "https://gj491-mk2w3yrm-eastus2.cognitiveservices.azure.com/"
temperature = 0
api_v = "2025-01-01-preview"


llm = load_llm(credentials,end_p,model_id,temperature,api_v)
df = extract_data("data/btcusd_1-min_data.csv")
graph = AgentsGraph(llm,"pdf")
app = graph.build_workflow()


while True:
    print("prompt>> ",end="")
    q = input()
    initial_state = AgentState(
        question = q,
        decision = "",
        analysis = "",
        report = "",
        dataframe = df
    )
    graph.run_work_flow(app,initial_state)
    print("\n")


    while True:
    print("prompt>> ",end="")
    q = input()
    initial_state = AgentState(
        question = q,
        decision = "",
        analysis = "",
        report = "",
        dataframe = df
    )
    graph.run_work_flow(app,initial_state)
    print("\n")