import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from models.schema import RouteSchema, DataAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst


llm = pick_llm("advanced")

llm_router = llm.with_structured_output(RouteSchema)


# ---------------------------- DATA AGENT GRAPH ---------------------------- #


def router_node(state:DataAgentSchema):

    message = state.messages[-1].content

    route_response_dict = llm_router.invoke(message).model_dump()

    return {"route_response": route_response_dict['answer']}

def etl_node(state:DataAgentSchema):

    message = state.messages[-1].content

    response = etl_analyst.invoke({"messages": [HumanMessage(content=message)]})

    # Return only the ETL agent's final message: the `add` reducer appends it
    return {"messages": [response["messages"][-1]]}

def sql_node(state:DataAgentSchema):

    message = state.messages[-1].content

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

    return {"messages": [AIMessage(content=response["final_answer"])]}


data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")

def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "sql":
        return "sql_node"
    elif state.route_response == "etl":
        return "etl_node"
    else:
        raise ValueError(f"Invalid route response: {state.route_response}")


data_agent_graph.add_conditional_edges("router_node", route_edge,
                                    {
                                        "sql_node": "sql_node",
                                        "etl_node": "etl_node"
                                    })

data_agent = data_agent_graph.compile()


if __name__ == "__main__":

    # Optional|
    from IPython.display import display, Image
    img = Image(data_agent.get_graph().draw_mermaid_png())

    with open("data_agent_graph.png", "wb") as f:
        f.write(img.data)

    response = data_agent.invoke(
        {
            "messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extracted folder in the csv folder")],
            "route_response": ""
        }
    )

    print(response)