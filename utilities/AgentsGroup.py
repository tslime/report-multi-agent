import pandas as pd

from models.AgentState import AgentState
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent 
from fpdf import FPDF

def load_llm(api_k,end_point,model,temp,api_v):
    
    #Load llm with langchain tools
    loaded_llm = AzureChatOpenAI(
    api_key=api_k,
    azure_endpoint=end_point,
    deployment_name=model,
    temperature=temp,
    streaming=False,
    api_version=api_v
    )

    return loaded_llm



def supervisor_agent_node(state:AgentState,language_model):
    
    engineered_prompt = f"""
    You are supervisor agent. Your tasks is to decide whether the user's question require data analysis. 

    user question: {state["question"]}

    If the question is about analyzing data, statistics, trends, or queries on a dataset, respond with:  data_agent
    If the question is simple and doesn't need data analysis, respond with: direct_response

    Respond with one word only: data_agent or direct_response
    """

    supervisor_agent = create_pandas_dataframe_agent(
    language_model,
    pd.DataFrame(),
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
    agent_type="openai-tools"
    )

    supervisor_result = supervisor_agent.invoke(engineered_prompt,handle_parsing_errors=True)
    return {"decision":supervisor_result["output"]}


def direct_response_node(state:AgentState,language_model):
    response = language_model.invoke(state["question"])
    return {"report":response.content}



def data_agent_node(state:AgentState,language_model):
    engineered_prompt = f"""
    You are a Python data analysis expert.

    Given the user's question: "{state['question']}"

    If the user's request involves visualizing data (for example, showing a trend, distribution, or other chart), generate executable Python code 
    to create the appropriate chart from the pandas DataFrame 'df'. Save the plot as 'generated-reports/btc_trend.png'.

    If no chart is needed, simply analyze the data and return findings as text.

    Return ONLY the Python code for the chart if required, otherwise return only a textual analysis.
    """
    
    report_agent = create_pandas_dataframe_agent(
    language_model,
    state["dataframe"],
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
    agent_type="openai-tools"
    )

    answers = report_agent.invoke(engineered_prompt,handle_parsing_errors=True)
    return {"analysis":answers["output"]}

def report_writer_agent_node(state:AgentState,language_model,report_format):
    
    engineered_prompt = f"""
    You are an expert report writer.

    original question: {state["question"]}
    analysis result: {state["analysis"]}
    report_format : {report_format}

    If a chart image (e.g., 'btc_trend.png') has been generated, embed it in the report using the appropriate syntax:
    - For Markdown, use ![Bitcoin Trend](btc_trend.png) in a 'Visualization' section.
    
    If no chart exists, omit the visualization section.

    Base your report on the provided analysis.
    """

    writer_agent = create_pandas_dataframe_agent(
    language_model,
    pd.DataFrame(),
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
    agent_type="openai-tools"
    )

    writer_result = writer_agent.invoke(engineered_prompt,handle_parsing_errors=True)

    if report_format == "md":
        with open("generated-reports/statistical-report.md","w",encoding="utf-8") as f:
            f.write(writer_result["output"])

    elif report_format == "pdf":
        write_to_pdf(writer_result["output"])
        
    return {"report":writer_result["output"]}


def route_decision(state: AgentState):
    return state["decision"]


def write_to_pdf(analysis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times",size=12)

    for line in analysis.split("\n"):
        pdf.cell(0,10,line,ln=True)

    pdf.output("generated-reports/statistical-report.pdf")