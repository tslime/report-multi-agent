import os
import sys
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent 


load_dotenv()

def supervisor_agent(user_question:str):
    
    engineered_prompt = f"""
    You are supervisor agent. Your tasks is to decide whether the user's question require data analysis. 

    user question: {user_question}

    If the question is about analyzing data, statistics, trends, or queries on a dataset, respond with:  data_agent
    If the question is simple and doesn't need data analysis, respond with: direct

    Respond with one word only: data_agent or direct
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
    return supervisor_result["output"]


def data_agent(question:str,dataframe):
    report_agent = create_pandas_dataframe_agent(
    llm,
    dataframe,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
    agent_type="openai-tools"
    )

    answers = report_agent.invoke(prompt,handle_parsing_errors=True)
    return answers["output"]

def report_writer_agent(original_question:str,analysis:str):
    
    engineered_prompt = f"""
    You are an expert report writer.

    original question: {original_question}
    analysis result: {analysis}

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
    return writer_result["output"]


#Model config
credentials = os.getenv("API_KEY")
model_id = "gpt-4o-mini"
parameters = {
    "temperature": 0
}



#Load llm with langchain tools
llm = AzureChatOpenAI(
    api_key=credentials,
    azure_endpoint="https://gj491-mk2w3yrm-eastus2.cognitiveservices.azure.com/",
    deployment_name=model_id,
    temperature=parameters["temperature"],
    streaming=False,
    api_version="2025-01-01-preview"
    )

#Load data set
df = pd.read_csv("data/btcusd_1-min_data.csv")


#Intiate agent for querying data set



#Start agent
while True:
    print("Prompt>> ",end="")
    q = input()
    
    print("Answer: ",answers["output"])
    print("\n")