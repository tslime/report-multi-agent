import os
import sys
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent 
from utilities.AgentState import AgentState
from langgraph.graph import StateGraph, END

load_dotenv()

def supervisor_agent_node(state:AgentState):
    
    engineered_prompt = f"""
    You are supervisor agent. Your tasks is to decide whether the user's question require data analysis. 

    user question: {state["question"]}

    If the question is about analyzing data, statistics, trends, or queries on a dataset, respond with:  data_agent
    If the question is simple and doesn't need data analysis, respond with: direct_response

    Respond with one word only: data_agent or direct_response
    """

    supervisor_agent = create_pandas_dataframe_agent(
    llm,
    pd.DataFrame(),
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
    agent_type="openai-tools"
    )

    supervisor_result = supervisor_agent.invoke(engineered_prompt,handle_parsing_errors=True)
    return {"decision":supervisor_result["output"]}


def direct_response_node(state:AgentState):
    response = llm.invoke(state["question"])
    return {"report":response.content}



def data_agent_node(state:AgentState):
    report_agent = create_pandas_dataframe_agent(
    llm,
    state["dataframe"],
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
    agent_type="openai-tools"
    )

    answers = report_agent.invoke(state["question"],handle_parsing_errors=True)
    return {"analysis":answers["output"]}

def report_writer_agent_node(state:AgentState):
    
    engineered_prompt = f"""
    You are an expert report writer.

    original question: {state["question"]}
    analysis result: {state["analysis"]}

    Write a clear, well redacted, and formatted report on the basis of this analysis.
    """

    writer_agent = create_pandas_dataframe_agent(
    llm,
    pd.DataFrame(),
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
    agent_type="openai-tools"
    )

    writer_result = writer_agent.invoke(engineered_prompt,handle_parsing_errors=True)
    return {"report":writer_result["output"]}








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

#Load data set
df = pd.read_csv("data/btcusd_1-min_data.csv")



#Start agent
while True:
    print("Prompt>> ",end="")
    q = input()
    
    print("Answer: ",answers["output"])
    print("\n")


def route_decision(state: AgentState):
    return state["decision"]

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