"""
ROUTER (ORCHESTRATOR) AGENT
Data Agent: the single entry point. It reads the user's request, decides which
specialist agent should handle it, and passes the request on.

  - "sql" -> SQL Analyst  (questions about the data in Postgres)
  - "etl" -> ETL Analyst  (download data from an API, or transform a file)

This is the "router" pattern of multi-agent systems: one small, cheap decision
up front, then each specialist does what it's good at. Adding a new skill
later means adding one more branch, not changing the specialists.

Flow:

    START
      -> router_node   (LLM, advanced, structured output)  sql or etl?
           |-- "sql" -> sql_node   run the whole SQL analyst graph
           |-- "etl" -> etl_node   run the whole ETL analyst graph
      -> END

The SQL and ETL analysts are full LangGraph graphs of their own. Here each one
is called from inside a node, so they work as *sub-graphs*: the data agent
doesn't know (or care) how many steps they take inside.

Run it with:  uv run agents/data_agent.py   (or uv run main.py)
It also saves a picture of the graph to img/data_agent_graph.png.
"""

import os
import sys

# Let this script import from the project root (utils/, models/, agents/).
# See the step-by-step explanation at the top of sql_analyst.py.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Project imports ----------------------------------------------------------
from utils.llm_pick import pick_llm                       # chooses the LLM by cost level
from models.schema import RouteSchema, DataAgentSchema    # router output shape + graph state

# --- LangChain / LangGraph ------------------------------------------------------
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

# --- The specialist agents (already compiled graphs) ----------------------------
# Importing them runs their module code, which builds and compiles each graph.
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst


# The router's LLM. with_structured_output(RouteSchema) forces the reply into
# RouteSchema's shape, so `answer` is always exactly "sql" or "etl"
# (it's a Literal), never free text like "I think this is about SQL".
llm = pick_llm("advanced")

llm_router = llm.with_structured_output(RouteSchema)


# ======================================================================================
# NODES
# Same rule as the other agents: each node returns a dict with ONLY the fields
# it changed. `messages` uses the `add` reducer, so new messages are appended.
# ======================================================================================

# ---------------------------- DATA AGENT GRAPH ---------------------------- #

# --------------------------------------------------------------------------------------
# Node 1: router_node  (decide)
# --------------------------------------------------------------------------------------
def router_node(state:DataAgentSchema):
    """Classify the latest user message as "sql" or "etl"."""

    # The latest message is the user's request.
    message = state.messages[-1].content

    # The reply is a RouteSchema object; model_dump() turns it into a dict:
    # {"answer": "etl", "comments": "The user wants to download data from an API ..."}
    route_response_dict = llm_router.invoke(message).model_dump()

    # Only store the decision; the conditional edge below reads it.
    return {"route_response": route_response_dict['answer']}

# --------------------------------------------------------------------------------------
# Node 2a: etl_node  (hand off to the ETL analyst)
# --------------------------------------------------------------------------------------
def etl_node(state:DataAgentSchema):
    """Run the ETL analyst graph on the user's request."""

    message = state.messages[-1].content

    # Run the whole ETL sub-graph. It starts with a fresh history containing only
    # this request, and returns its final state: {"messages": [...all its messages...]}
    response = etl_analyst.invoke({"messages": [HumanMessage(content=message)]})

    # Return only the ETL agent's final message: the `add` reducer appends it
    return {"messages": [response["messages"][-1]]}

# --------------------------------------------------------------------------------------
# Node 2b: sql_node  (hand off to the SQL analyst)
# --------------------------------------------------------------------------------------
def sql_node(state:DataAgentSchema):
    """Run the SQL analyst graph on the user's request."""

    message = state.messages[-1].content

    # The SQL analyst's state (AgentSchema) has many required fields, so every
    # one needs a starting value. The user's request goes in `user_input`;
    # the rest start empty and are filled in by its nodes.
    # is_safe starts as "No" (a valid Literal value); the judge node overwrites it.
    input_schema = {
        "messages": [],
        "user_input": message,
        "curated_question": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    response = sql_analyst.invoke(input_schema)

    # The SQL analyst keeps its answer in `final_answer` (a str). Wrap it in an
    # AIMessage so both branches add the same kind of thing to the history.
    return {"messages": [AIMessage(content=response["final_answer"])]}


# ======================================================================================
# GRAPH
# ======================================================================================

data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

# Every request goes to the router first.
data_agent_graph.add_edge(START, "router_node")

# Router function for the conditional edge: turn the stored decision into the
# name of the next node. Raising an error on anything else makes a bad value
# fail loudly instead of silently going down the wrong branch.
def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "sql":
        return "sql_node"
    elif state.route_response == "etl":
        return "etl_node"
    else:
        raise ValueError(f"Invalid route response: {state.route_response}")


# The dict maps what route_edge returns -> the node to run.
data_agent_graph.add_conditional_edges("router_node", route_edge,
                                    {
                                        "sql_node": "sql_node",
                                        "etl_node": "etl_node"
                                    })

# Note: there's no add_edge("sql_node", END) / ("etl_node", END). LangGraph
# finishes the run when a node has no outgoing edge, so both branches end there.

# Compile at module level so main.py can import `data_agent`.
data_agent = data_agent_graph.compile()


# ======================================================================================
# RUN  (only when this file is executed directly, not when it's imported)
# ======================================================================================
if __name__ == "__main__":

    # Save a diagram of the graph to img/.
    # Optional|
    from IPython.display import display, Image
    img = Image(data_agent.get_graph().draw_mermaid_png())

    with open(os.path.join(os.path.dirname(__file__), "..", "img", "data_agent_graph.png"), "wb") as f:
        f.write(img.data)

    # Starting state: the user's request as the first message, and an empty
    # route_response (required field) that router_node fills in.
    # Try a SQL question too, e.g.
    #   "What are the different types of payment methods we have in our database?"
    response = data_agent.invoke(
        {
            "messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extracted folder in the csv folder")],
            "route_response": ""
        }
    )

    # Final state: {"messages": [request, answer], "route_response": "etl"}
    print(response)
