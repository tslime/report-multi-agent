# Multi-Agent Data Analysis System with LangGraph

A sophisticated multi-agent system built with LangGraph that intelligently routes user questions through specialized AI agents for data analysis and report generation.

## Overview

This application demonstrates a production-ready multi-agent architecture where:
1. A **Supervisor Agent** determines if a question requires data analysis
2. A **Data Analysis Agent** performs statistical analysis on CSV datasets
3. A **Report Writer Agent** generates formatted, professional reports
4. **Direct Response** handles simple questions without data processing

The system uses Azure OpenAI and LangGraph for orchestration, showcasing advanced agentic workflow patterns.

## Architecture

```
User Question
     ↓
Supervisor Agent (Router)
     ↓
   ┌─────────────┬─────────────┐
   ↓             ↓             ↓
Direct      Data Agent    (rejected)
Response         ↓
   ↓      Report Writer
   ↓             ↓
   └─────────────┘
         ↓
   Final Answer
```

## Features

- 🤖 **Multi-Agent System**: Specialized agents for different tasks
- 🔀 **Intelligent Routing**: Supervisor agent decides workflow path
- 📊 **Data Analysis**:  Pandas-powered statistical analysis
- 📝 **Report Generation**: Professional formatted output
- 🔄 **State Management**: LangGraph state machine for reliable execution
- 💬 **Interactive CLI**: Simple chat interface
- ⚡ **Azure OpenAI**:  Enterprise-grade AI capabilities

## Prerequisites

- Python 3.9+
- Azure OpenAI API access
- Dataset in CSV format (default:  Bitcoin 1-minute data)

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. **Install dependencies**
```bash
pip install langchain langchain-openai langchain-experimental langgraph pandas python-dotenv
```

3. **Set up environment variables**

Create a `.env` file:
```env
API_KEY=your_azure_openai_api_key_here
```

4. **Add your dataset**

Place your CSV file in the `data/` directory:
```
data/your_dataset.csv
```

Update the path in the code:
```python
df = pd.read_csv("data/your_dataset.csv")
```

## Project Structure

```
.
├── main.py                     # Main application with LangGraph
├── .env                        # Environment variables (not committed)
├── data/
│   └── your_dataset.csv       # Your dataset (not committed)
├── .gitignore                 # Git ignore file
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Agent Architecture

### 1. Supervisor Agent
**Purpose**: Routes questions to appropriate handlers

**Input**: User question  
**Output**: `data_agent` or `direct`  
**Model**: GPT-4o-mini (lightweight, fast)

```python
def supervisor_agent(user_question: str) -> str:
    # Decides:  Does this need data analysis?
    # Returns: "data_agent" or "direct"
```

### 2. Data Analysis Agent
**Purpose**: Performs statistical analysis on datasets

**Input**: User question + dataframe  
**Output**: Analysis results  
**Model**: GPT-4o-mini with pandas agent  
**Capabilities**:  SQL-like queries, statistics, aggregations, filtering

```python
def data_agent(question: str, dataframe) -> str:
    # Executes pandas operations
    # Returns: Raw analysis results
```

### 3. Report Writer Agent
**Purpose**:  Formats analysis into professional reports

**Input**: Original question + analysis results  
**Output**: Formatted report  
**Model**: GPT-4o-mini  
**Features**: Clear formatting, context-aware, actionable insights

```python
def report_writer_agent(original_question: str, analysis: str) -> str:
    # Transforms raw analysis into polished report
    # Returns: Human-readable formatted report
```

### 4. Direct Response Handler
**Purpose**:  Answers simple questions without data processing

**Input**: User question  
**Output**:  Direct answer  
**Use case**:  Greetings, general questions, metadata queries

## LangGraph Workflow

The system uses a **StateGraph** to manage agent execution:

```python
class AgentState(TypedDict):
    question: str       # User's original question
    decision: str       # Supervisor's routing decision
    analysis: str       # Data agent's results
    report: str         # Final formatted output
    dataframe: object   # The dataset
```

**Node Flow:**
1. `START` → Supervisor Node
2. Supervisor Node → (conditional routing)
   - If "data_agent" → Data Agent Node → Report Writer Node → `END`
   - If "direct" → Direct Response Node → `END`

## Usage

### Running the Application

```bash
python main.py
```

### Example Interactions

**Data Analysis Questions:**
```
Prompt>> What is the average price in the dataset? 

[Supervisor routes to data_agent]
[Data agent analyzes]
[Report writer formats]

Answer:  Based on the analysis of the dataset, the average price is $45,231.50... 
```

**Simple Questions:**
```
Prompt>> What is Bitcoin? 

[Supervisor routes to direct response]

Answer: Bitcoin is a decentralized digital currency...
```

**Statistical Queries:**
```
Prompt>> Find the correlation between volume and price

[Full multi-agent pipeline executes]

Answer: Analysis reveals a correlation coefficient of 0.67...
```

## Configuration

### Model Settings

```python
model_id = "gpt-4o-mini"
parameters = {
    "temperature": 0  # Deterministic responses
}
```

### Azure OpenAI Setup

```python
llm = AzureChatOpenAI(
    api_key=credentials,
    azure_endpoint="your-endpoint",
    deployment_name=model_id,
    api_version="2025-01-01-preview"
)
```

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **LangGraph** | Agent orchestration and state management |
| **LangChain** | LLM framework and tooling |
| **Azure OpenAI** | GPT-4o-mini language model |
| **Pandas** | Data manipulation and analysis |
| **Python-dotenv** | Environment configuration |

## Advanced Features

### State Management
LangGraph maintains state across agent transitions, ensuring: 
- Context preservation
- Error recovery
- Execution traceability

### Conditional Routing
Supervisor agent dynamically routes based on question complexity:
```python
def route_decision(state: AgentState) -> str:
    if "data_agent" in state["decision"]:
        return "data_agent"
    else:
        return "direct_response"
```

### Extensibility
Easy to add new agents: 
1. Define agent function
2. Create node wrapper
3. Add to graph
4. Update routing logic

## Security & Best Practices

**Security Considerations:**

- Uses `allow_dangerous_code=True` for pandas agent execution
- Only use with trusted datasets
- Review generated code in verbose mode
- Never commit `.env` file with API keys
- Validate user inputs in production

**Best Practices:**
- Keep API keys in environment variables
- Use separate keys for dev/prod
- Monitor API usage and costs
- Implement rate limiting for production
- Add error handling and logging

## Troubleshooting

**Issue**:  `NameError: name 'llm' is not defined`
- **Cause**: LLM initialization after agent definitions
- **Solution**: Move LLM setup before agent functions

**Issue**: `KeyError: 'output'`
- **Cause**: Agent invoke returning unexpected format
- **Solution**: Check `return_intermediate_steps=True` and access `answers["output"]`

**Issue**:  Supervisor always routes to direct
- **Solution**: Improve supervisor prompt engineering

**Issue**: Data agent fails silently
- **Solution**: Enable `verbose=True` and check pandas operations

## Roadmap

- [ ] Add memory/conversation history
- [ ] Support multiple datasets
- [ ] Add visualization agent for charts/graphs
- [ ] Implement streaming responses
- [ ] Add evaluation metrics
- [ ] Support other LLM providers
- [ ] Add web interface
- [ ] Implement agent reflection/self-correction

## Performance Considerations

- **Supervisor**: ~1-2 seconds per request
- **Data Agent**: 3-10 seconds (depends on query complexity)
- **Report Writer**:  ~2-3 seconds
- **Total**: 6-15 seconds for full pipeline

**Optimization Tips:**
- Use cheaper models for supervisor
- Cache common queries
- Implement parallel execution where possible
- Use streaming for better UX

## Contributing

Contributions welcome! Areas of interest:
- New specialized agents
- Alternative routing strategies
- Performance optimizations
- Additional LLM provider support

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by Azure OpenAI
- Inspired by multi-agent research from OpenAI, Anthropic, and Google DeepMind

## Example Use Cases

- **Financial Analysis**: Query trading data, find patterns
- **Sales Data**: Analyze trends, forecast, segment customers
- **Scientific Data**: Statistical analysis, correlation studies
- **Operational Metrics**: Performance analysis, anomaly detection
- **Survey Data**: Sentiment analysis, response aggregation

## Citation

If you use this project in research: 
```bibtex
@software{multiagent_langgraph,
  author = {tslimeoh},
  title = {Multi-Agent Data Analysis System with LangGraph},
  year = {2026},
  url = {https://github.com/tslimeoh/your-repo}
}
```

---

**Questions?  Issues?** Open an issue on GitHub! 