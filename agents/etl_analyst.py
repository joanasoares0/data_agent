"""
TOOL-CALLING (ReAct) AGENT
ETL Analyst Agent: ask in plain English to extract or transform data, and the
agent picks the right tool and runs it.

ETL = Extract, Transform, Load:
  - Extract:   get data from somewhere (here: a web API)
  - Transform: change it (filter, clean, aggregate, ...)
  - Load:      save the result somewhere (here: a CSV / JSON / Parquet file)

How it differs from the SQL analyst
-----------------------------------
The SQL analyst is a *workflow*: the path through the graph is fixed in code
(curate -> context -> SQL -> judge -> run -> answer).

This agent is a *ReAct loop* (Reason + Act): the LLM itself decides what to do
next. It looks at the conversation and the list of tools it has, and either
  - asks to call a tool (Act), then sees the tool's result and thinks again, or
  - answers the user, which ends the loop.

Flow:

    START
      -> llm_node      (LLM, advanced)  read history, decide: call a tool or answer?
           |-- tool call  -> tool_node  run the tool(s), add the result to history
           |                   -> llm_node  (loop back: the LLM sees the result)
           |-- no tool call -> END      the LLM's message is the final answer

Tools:
  - extract_load_tool:   download JSON from an API and save it as a file
  - transform_load_tool: ask an LLM to write pandas code for the transformation,
                         then run that code

Run it with:  uv run agents/etl_analyst.py
It also saves a picture of the graph to img/etl_analyst_graph.png.
"""

import os
import sys

# Let this script import from the project root (utils/, models/).
# See the step-by-step explanation at the top of sql_analyst.py.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Project imports ----------------------------------------------------------
from utils.llm_pick import pick_llm          # chooses the LLM by cost level
from utils.etl_tools import ETLTools         # the real work: download, read, run code
from models.schema import ETLAgentSchema     # graph state: just the message history

# --- LangChain / LangGraph ------------------------------------------------------
# Message types that make up the conversation history:
#   HumanMessage -> what the user said
#   AIMessage    -> what the LLM said (may contain tool_calls)
#   ToolMessage  -> the result of running a tool, linked to the call by tool_call_id
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
# @tool turns a normal Python function into a LangChain "tool": an object with a
# name, a description (the docstring) and an argument schema (the type hints).
from langchain.tools import tool


# ======================================================================================
# TOOLS
# What the LLM can "do". The LLM never runs these itself: it only asks for a
# call (name + arguments), and tool_node below runs the real Python function.
#
# The docstring and type hints matter: they are sent to the LLM so it knows
# what each tool does and which arguments to pass. A vague docstring means
# the LLM picks the wrong tool or wrong arguments.
# ======================================================================================

# --------------------------------------------------------------------------------------
# Tool 1: extract_load_tool  (Extract + Load)
# --------------------------------------------------------------------------------------
@tool
def extract_load_tool(url:str, output_folder:str, format:str) -> str:
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    # The tool is a thin wrapper: the logic lives in utils/etl_tools.py, so it
    # can be tested without an LLM.
    etl_tools = ETLTools()
    return etl_tools.extract_load(url, output_folder, format)


# --------------------------------------------------------------------------------------
# Tool 2: transform_load_tool  (Transform + Load)
# --------------------------------------------------------------------------------------
@tool
def transform_load_tool(
    input_file_path:str,
    output_folder:str,
    output_format:str, 
    user_question:str
    ) -> str:
    """
    This tool transforms the data from the specified file and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()

    # Step 1: give the LLM context about the data.
    # Like the sample rows in the SQL agent: the LLM writes better code when it
    # sees the real column names and example values.
    top_3_rows = etl_tools.transform_load_context(input_file_path)

    # Step 2: a second LLM, *inside* the tool, writes the pandas code.
    llm = pick_llm("advanced")

    # Prompt: role + task + "only code, no explanation" (the reply is executed
    # as-is) + where to read/save + the user's request + the data sample.
    prompt = f"""
        You are a Python Data Analyst who uses Pandas to analyze data. 
        You need to provide only the Pandas Code that will help to perform the right ETL operations on the data stored in the file : {input_file_path} as per the user's question. 
        Do not provide any explanation or comments, only the code should be provided. 
        The code should be in a format that can be executed in a Python environment with Pandas installed. 
        Don't write anything else than Pandas Code. \n
        
        Create the Pandas Dataframe from the data stored in the file : {input_file_path} and then write the code to transform and save the data at {output_folder}.
        Here's the user's question: {user_question}\n
        Here's the context of the data you will be analyzing: {top_3_rows}\n
        """

    # .text (not .content): some models return .content as a LIST of blocks
    # ([{"type": "text", "text": "..."}]) instead of a plain string.
    # .text always gives the reply as one string.
    response = llm.invoke(prompt).text

    # Step 3: clean the reply. LLMs often wrap code in a Markdown fence:
    #   ```python
    #   import pandas as pd ...
    #   ```
    # strip('```') removes the backticks, lstrip('python') removes the language tag.
    # Optional Cleaning
    pandas_code = response.strip().strip('```').strip().lstrip('python').strip()

    # Step 4: run the generated code. It reads the input file and writes the output file.
    # ⚠ exec() runs LLM-written code with full access to this machine.
    #   Fine for learning locally, never in production (see README).
    # Execute the Pandas code
    results = etl_tools.execute_code(pandas_code)

    # The returned text becomes a ToolMessage the agent LLM reads in its next turn.
    return f"The data is transformed and saved at {output_folder} in {output_format} format. \n\n Pandas Code Executed: \n {pandas_code} \n\n Execution Result: \n {results}"


# --------------------------------------------------------------------------------------
# Toolkit + the agent's LLM
# --------------------------------------------------------------------------------------
# The list of tools the agent is allowed to use.
# Toolkit 
tools = [extract_load_tool, transform_load_tool]

# bind_tools() attaches the tools' names, descriptions and argument schemas to
# every call. The LLM can then answer with a normal message OR with
# `tool_calls`, e.g. [{"name": "extract_load_tool", "args": {...}, "id": "call_1"}].
llm = pick_llm("advanced")
llm_bind = llm.bind_tools(tools)


# ======================================================================================
# GRAPH
# Two nodes in a loop: llm_node (think) <-> tool_node (act).
# ======================================================================================

# --------------------------------------------------------------------------------------
# Node 1: llm_node  (Reason)
# --------------------------------------------------------------------------------------
def llm_node(state:ETLAgentSchema):
    """
    Ask the LLM what to do next, given the whole conversation so far.

    The reply is an AIMessage that either has tool_calls (-> go to tool_node)
    or doesn't (-> it's the final answer, go to END).
    """

    messages = state.messages

    # The chat history (user request + earlier tool calls + tool results) is
    # pasted into the prompt, so on the second turn the LLM sees "the tool
    # succeeded" and knows it can stop.
    prompt = f"""
            You are a Python Data Analyst who has access to tools that can extract and load, 
            transform and load data. You will be provided with a user's question 
            and you would need to perform the right ETL operations as per the user's question. 
            If the operation is performed then inform the user and end the coversation.
            Here's the chat history: {messages}\n
    """

    final_answer = llm_bind.invoke(prompt)

    # Return only the new message: the `add` reducer appends it to the history
    return {"messages": [final_answer]}


# --------------------------------------------------------------------------------------
# Node 2: tool_node  (Act)
# --------------------------------------------------------------------------------------
def tool_node(state:ETLAgentSchema):
    """
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    """

    tools_results = []

    # Lookup table: tool name -> tool object, e.g. {"extract_load_tool": <tool>, ...}
    tools_by_name = {tool.name: tool for tool in tools}

    # The last message is the LLM's reply from llm_node; it holds the requested calls.
    tool_calls = state.messages[-1].tool_calls

    # The LLM can ask for several calls at once, so run each one.
    for tool_call in tool_calls:

        tool = tools_by_name[tool_call['name']]
        # tool_call['args'] is a dict like {"url": "...", "output_folder": "...", "format": "csv"}
        observation = tool.invoke(tool_call['args'])

        # Wrap the result in a ToolMessage. tool_call_id links it to the request,
        # which the OpenAI API requires (every tool call needs a matching result).
        tools_results.append(ToolMessage(content=observation, tool_call_id = tool_call['id']))

    # Only the new ToolMessages: the `add` reducer appends them to the history.
    return {"messages": tools_results}


# --------------------------------------------------------------------------------------
# Nodes & Edges
# --------------------------------------------------------------------------------------
# Nodes & Edges
etl_analyst_graph = StateGraph(ETLAgentSchema)

# Here the node ID is given explicitly as the first argument ("llm_node").
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

# Always start by asking the LLM.
etl_analyst_graph.add_edge(START, "llm_node")

# Router: after llm_node, did the LLM ask for a tool?
# Returns a *key* ("tool_node" or "end"); the dict below maps keys to nodes.
def is_tool_call(state:ETLAgentSchema):
    tool_calls = state.messages[-1].tool_calls

    # An empty list is "falsy" in Python, so `if tool_calls:` means "at least one call".
    if tool_calls:
        return "tool_node"
    else:
        return "end"

etl_analyst_graph.add_conditional_edges(
    "llm_node",is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END
    }
)

# After running tools, always go back to the LLM: this edge is what makes it a loop.
etl_analyst_graph.add_edge("tool_node", "llm_node")

# Compile at module level so other agents (data_agent.py) can import `etl_analyst`.
etl_analyst = etl_analyst_graph.compile()


# ======================================================================================
# RUN  (only when this file is executed directly, not when it's imported)
# ======================================================================================
if __name__ == "__main__":
    #llm_bind = pick_llm("advanced").bind_tools(tools)
    #print(llm.invoke("I want to extract the data from the API endpoint 'https://api.example.com/data' and save it as a csv file in the folder '/data/output'."))


    # Save a diagram of the graph to img/ (path built from this file's folder,
    # so it works no matter where you run the script from).
    # Optional
    from IPython.display import display, Image
    img = Image(etl_analyst.get_graph().draw_mermaid_png())
    with open(os.path.join(os.path.dirname(__file__), "..", "img", "etl_analyst_graph.png"), "wb") as f:
        f.write(img.data)

    # Example 1 (extract): download the Pokémon list and save it as CSV.
    #response = etl_analyst.invoke(
    #    {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extracted folder in the csv folder")]}
    #)

    # Example 2 (transform): filter the extracted file. Run example 1 first.
    # The paths are relative to the project root, so run from there.
    response = etl_analyst.invoke(
         {"messages":[HumanMessage(content="""
            I want to transform the data stored in the 'data/extracted/extracted_data.csv' file 
            and save the transformed data in the 'data/transform' folder in the csv format.
            The transformation should filter the data to show bulbasaur pokemon only.
    """)]}
    )    

    # The final state: the full message history (request, tool call, tool result, answer).
    print(response)
