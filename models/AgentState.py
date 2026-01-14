import os
import sys

from typing import TypedDict

class AgentState(TypedDict):
    question: str
    decision: str
    analysis: str
    report: str
    dataframe: object