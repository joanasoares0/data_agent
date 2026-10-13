"""
Entry point of the project: send one request to the Data Agent.

The Data Agent (agents/data_agent.py) decides whether the request is about
SQL (questions about the Postgres data) or ETL (download / transform files),
and hands it to the right specialist agent.

Run it from the project root:  uv run main.py
Change the text in HumanMessage(...) below to ask something else.
"""

# No sys.path trick needed here: main.py is already in the project root, so
# Python can find the agents/ folder directly.
from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage

if __name__ == "__main__":

    # The starting state of the Data Agent (DataAgentSchema):
    #   messages:       the conversation so far; here just the user's request
    #   route_response: required field, starts empty; the router fills in "sql" or "etl"
    response = data_agent.invoke(
        {
            "messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
            "route_response": ""
         }
    )

    # The final state. The answer is the last message: response["messages"][-1]
    print(response)
