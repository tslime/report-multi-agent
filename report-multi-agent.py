import os
import sys


from dotenv import load_dotenv
from functools import partial
from models.AgentState import AgentState
from langgraph.graph import StateGraph, END
from utilities.DataManager import extract_data
from utilities.AgentsGroup import load_llm, supervisor_agent_node, data_agent_node, report_writer_agent_node, direct_response_node, route_decision


load_dotenv()


#Model configs
credentials = os.getenv("API_KEY")
model_id = "gpt-4o-mini"
end_p = "https://gj491-mk2w3yrm-eastus2.cognitiveservices.azure.com/"
temperature = 0
api_v = "2025-01-01-preview"


llm = load_llm(credentials,end_p,model_id,temperature,api_v)
df = extract_data("data/btcusd_1-min_data.csv")



#Graph
workflow = StateGraph(AgentState)

workflow.add_node("supervisor",partial(supervisor_agent_node,language_model=llm))
workflow.add_node("direct_response",partial(direct_response_node,language_model=llm))
workflow.add_node("data_agent",partial(data_agent_node,language_model=llm))
workflow.add_node("report_writer",partial(report_writer_agent_node,language_model=llm,report_format="md"))


#Agents reasoning workflow
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
"supervisor",
route_decision,
    {
        "data_agent":"data_agent",
        "direct_response":"direct_response"
    }
)
workflow.add_edge("data_agent","report_writer")
workflow.add_edge("report_writer",END)
workflow.add_edge("direct_response",END)


app = workflow.compile()


if __name__ == "__main__":
    initial_state = AgentState(
        question = "Analyze the correlation between Bitcoin price and trading volume over time. Provide a visualization and statistical summary.",
        decision = "",
        analysis = "",
        report = "",
        dataframe = df
    )
    app.invoke(initial_state)














"""
#Start agent
while True:
    print("Prompt>> ",end="")
    q = input()
    
    print("Answer: ",answers["output"])
    print("\n")
"""
