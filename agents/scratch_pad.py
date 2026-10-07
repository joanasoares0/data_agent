"""
Scratch pad: try one piece of the agent on its own.

This script tests the safety judge in isolation, without running the whole
graph. It's a quick way to experiment with the judge prompt and see what
with_structured_output returns.

Run it with:  uv run agents/scratch_pad.py
Expected output: something like
    Judge Response: {'is_safe': 'Yes', 'comments': 'The query only reads data ...'}

Try changing sql_query to "DROP TABLE users;" to see the judge answer 'No'.
"""

import os 
import sys

# Add the parent directory to the system path to allow imports from the parent directory
# (see the explanation at the top of sql_analyst.py).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from models.schema import AgentSchema, JudgeSchema
from utils.database import DatabaseUtil

from langchain_core.messages import HumanMessage

llm = pick_llm("medium")

# with_structured_output wraps the model so its reply is parsed into a JudgeSchema
# object (is_safe + comments) instead of free text.
llm_judge = llm.with_structured_output(JudgeSchema)

sql_query = "SELECT * FROM users WHERE id = 1;"

prompt = f"""
    You are an SQL Judge for data security. Your task is to determine whether the SQL query is safe or not. 
    The SQL query should only be used for data retrieval and should not modify the database in any way. 
    Neither the SQL query nor the prompt should contain any SQL commands that can modify the
    database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change
    the structure or content of the database. 
    If the SQL query is safe, respond with 'Yes' otherwise respond with 'No'. 
    Additionally, provide comments explaining your decision. 
    Here's the SQL query to evaluate:
    {sql_query}
    """

response = llm_judge.invoke(prompt).model_dump() # get the response as a dictionary instead of a Pydantic model instance with model dump

print("Judge Response:", response)
