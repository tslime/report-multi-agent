import os
import sys


from dotenv import load_dotenv

from models.AgentState import AgentState
from langgraph.graph import StateGraph, END
from utilities.DataManager import extract_data
from utilities.AgentsGroup import supervisor_agent_node, data_agent_node, report_writer_agent_node, direct_response_node, route_decision


load_dotenv()


#Model configs
credentials = os.getenv("API_KEY")
model_id = "gpt-4o-mini"
temperature = 0


#Load llm with langchain tools
llm = AzureChatOpenAI(
    api_key=credentials,
    azure_endpoint="https://gj491-mk2w3yrm-eastus2.cognitiveservices.azure.com/",
    deployment_name=model_id,
    temperature=temperature,
    streaming=False,
    api_version="2025-01-01-preview"
    )


df = extract_data("data/btcusd_1-min_data.csv")


"""
#Start agent
while True:
    print("Prompt>> ",end="")
    q = input()
    
    print("Answer: ",answers["output"])
    print("\n")
"""



#Graph
workflow = StateGraph(AgentState)

workflow.add_node("supervisor",supervisor_agent_node)
workflow.add_node("direct_response",direct_response_node)
workflow.add_node("data_agent",data_agent_node)
workflow.add_node("report_writer",report_writer_agent_node)


workflow.set_entry_point("supervisor")


workflow.set_conditional_edges(
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