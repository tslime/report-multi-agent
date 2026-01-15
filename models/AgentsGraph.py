import os 
import sys

from functools import partial
from models.AgentState import AgentState
from langgraph.graph import StateGraph, END
from utilities.AgentsGroup import supervisor_agent_node, data_agent_node, report_writer_agent_node, direct_response_node, route_decision

class AgentsGraph:

    def __init__(self,language_model,f):
        self.llm = language_model
        self.format = f
        self.workflow = None

    def build_workflow(self):

        self.workflow = self.assign_graph_nodes(self.workflow)
        self.workflow = self.assign_edges(self.workflow)
        graph = self.workflow.compile()

        return graph


    def assign_graph_nodes(self,wf):
        
        wf = StateGraph(AgentState)
        wf.add_node("supervisor",partial(supervisor_agent_node,language_model=self.llm))
        wf.add_node("direct_response",partial(direct_response_node,language_model=self.llm))
        wf.add_node("data_agent",partial(data_agent_node,language_model=self.llm))
        wf.add_node("report_writer",partial(report_writer_agent_node,language_model=self.llm,report_format=self.format))

        return wf
    
    def assign_edges(self,wf):
        
        wf.set_entry_point("supervisor")
        wf.add_conditional_edges(
        "supervisor",
        route_decision,
            {
            "data_agent":"data_agent",
            "direct_response":"direct_response"
            }
        )
        wf.add_edge("data_agent","report_writer")
        wf.add_edge("report_writer",END)
        wf.add_edge("direct_response",END)

        return wf
    
    def run_work_flow(self,app,ini_state:AgentState):
        app.invoke(ini_state)








