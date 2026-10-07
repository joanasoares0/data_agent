import os 
import sys

# Add the parent directory to the system path to allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
This line lets a Python script import modules from the folder one level above the one where the script lives.

Example:

project/
├── utils.py
└── scripts/
    └── run.pyg

Read it from the inside out:
__file__ is the path of the current script, e.g. project/scripts/run.py.
os.path.dirname(__file__) gives its folder: project/scripts.
os.path.join(..., '..') adds "go up one level": project/scripts/...
os.path.abspath(...) turns that into a clean full path: /home/you/project.
sys.path.append(...) adds that path to the list of places Python searches when you write import something.

Without the line, import utils inside run.py fails, because Python only looks in scripts/ (and installed packages). With the line, Python also looks in project/, so the import works.

Two notes:
The file needs import sys and import os at the top for this to run.
It works, but it's considered a workaround. Cleaner options are running the script as a module from the project root (python -m scripts.run) or making the project an installable package (pip install -e .). 
For a quick script it's fine; in a larger codebase it can make imports harder to follow.
"""

from utils.llm_pick import pick_llm
from models.schema import AgentSchema, JudgeSchema
from utils.database import DatabaseUtil

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from IPython.display import display, Image

# test if llm is working
#llm = pick_llm("low")
#print(llm.invoke("What is the capital of France?"))

# python functions are my nodes in the graph, and the edges are the calls between them. I can use this to build a graph of function calls and visualize it.

# ------------------------------------------------------- AI Agent code ----------------------------------------------------------------------------------

# 1st node
def curate_question(state: AgentSchema) -> dict:
    # Each node returns only the fields it changed, as a dict. LangGraph merges that into the state.
    # Don't return the whole state: `messages` uses the `add` reducer, so returning the full list
    # would append every existing message again (duplicates). Return just the new messages instead.
    """
    Curates a question based on the current state of the agent.

    Args:
        state (AgentSchema): The current state of the agent.

    Returns:
        str: The curated question.
    """
    
    user_question = state.user_input #this is a pydantic model object
    llm = pick_llm("low") #pick a low cost llm for this task, as it is not critical to have a high quality answer for this task.
    response = llm.invoke(f"Curate the following question: {user_question}").content

    # the add reducer appends this new message to the conversation history
    return {"curated_question": response, "messages": [HumanMessage(content=response)]}

# 2nd node
def prompt_query_context(state: AgentSchema) -> dict:
    """
    Generates a detailed prompt with SQL DB context that will help the agent to generate SQL query.

    Args:
        state (AgentSchema): The current state of the agent.

    Returns:
        str: The detailed prompt with SQL DB context.
    """

    curated_question = state.curated_question 

    connection_details = {
        "host": os.environ['POSTGRES_HOST'],
        "port": int(os.environ['POSTGRES_PORT']),
        "database": os.environ['POSTGRES_DB'],
        "user": os.environ['POSTGRES_USER'],
        "password": os.environ['POSTGRES_PASSWORD'],
    }
    
    obj = DatabaseUtil(connection_details)

    schema_info = obj.schema_details("rideshare") #get the schema details of the rideshare database, which is the default database for this agent.
    # we could also add the schema as a variable in models/schema.py, and then pass it to the agent as a parameter, but for now we will keep it simple and hardcode it here.

    # Constructing the prompt query for the agent to generate the SQL query
    prompt = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language 
    query into Postgres SQL query that can be executed on the database. 
    You are provided with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that 
    you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly asks for specific number of rows, always limit the output to 10 rows.
    All tables are in the "rideshare" schema, so always prefix table names with it (e.g. rideshare.payments).
    Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.  
    
    User's Original Query: 
    {curated_question}

    Database Schema Details:
    {schema_info}
    """    

    # update the prompt
    return {"prompt_query_context": prompt}


def generate_sql(state: AgentSchema) -> dict:
    """
    Generates an SQL query based on the curated question and prompt query context.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        str: The generated SQL query.
    """

    prompt = state.prompt_query_context

    llm = pick_llm("medium") #pick a medium cost llm for this task, as it is critical to have a high quality answer for this task.
    generated_sql_query = llm.invoke(prompt).content

    return {"generated_sql_query": generated_sql_query}


# Is safe node
def is_safe_sql(state: AgentSchema) -> dict:
    """
    Determines whether the generated SQL query is safe to execute or not.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        str: 'Yes' if the SQL query is safe to execute, 'No' otherwise  
    """
    sql_query = state.generated_sql_query

    llm = pick_llm("medium") #pick a medium cost llm for this task, as it is critical to have a high quality answer for this task.
    llm_judge = llm.with_structured_output(JudgeSchema)

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

    return {"is_safe": response['is_safe'], "comments": response['comments']}

def canceled_sql(state: AgentSchema) -> dict:
    """
    Cancels the SQL query execution if it is deemed unsafe.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        str: A message indicating that the SQL query execution has been canceled.
    """

    comments = state.comments
    final_answer = f"SQL query execution has been canceled due to safety concerns. Comments from the judge: {comments}"

    return {"final_answer": final_answer, "messages": [AIMessage(content=final_answer)]}

def execute_sql(state: AgentSchema) -> dict:
    """
    Executes the generated SQL query against the database.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        str: The result of executing the SQL query against the database.
    """

    sql_query = state.generated_sql_query

    connection_details = {
        "host": os.environ['POSTGRES_HOST'],
        "port": int(os.environ['POSTGRES_PORT']),
        "database": os.environ['POSTGRES_DB'],
        "user": os.environ['POSTGRES_USER'],
        "password": os.environ['POSTGRES_PASSWORD'],
        # The tables live in the rideshare schema, not public. Setting search_path makes
        # unqualified names like "FROM payments" resolve to rideshare.payments.
        "options": "-c search_path=rideshare,public",
    }

    obj = DatabaseUtil(connection_details)

    execution_result = obj.execute_query(sql_query)

    return {"sql_query_execution_result": execution_result}

def represent_final_answer(state: AgentSchema) -> dict:
    """
    Generates the final answer based on the SQL query execution result and the curated question.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        str: The final answer generated by the agent.
    """
    
    execution_result = state.sql_query_execution_result
    curated_question = state.curated_question

    llm = pick_llm("low") 
    prompt = f"""
        You are an SQL analyst agent. Your task is to provide a final answer to the user based on the
        execution result of the SQL query and the user's original question. 
        The final answer should be concise, clear, and directly address the user's query. 
        Avoid including any SQL code or technical details in the final answer. 
        The final answer should be in a user-friendly format that is easy to understand. 
        If the execution result is empty or does not provide a clear answer to the user's question, explain this in the final answer. \n
        Here is the execution result: 
        {execution_result} \n
        Here is the user's original question: 
        {curated_question}
        """

    llm_response = llm.invoke(prompt).content

    return {"final_answer": llm_response, "messages": [AIMessage(content=llm_response)]}


# create edges and nodes for the graph, where each node is a function and each edge is a call between functions.
sql_agent_graph = StateGraph(AgentSchema)

# nodes
sql_agent_graph.add_node(curate_question, name="Curate Question", description="Curates the user's question to make it more clear and specific.")
sql_agent_graph.add_node(prompt_query_context, name="Prompt Query Context", description="Generates a detailed prompt with SQL DB context that will help the agent to generate SQL query.")
sql_agent_graph.add_node(generate_sql, name="Generate SQL Query", description="Generates an SQL query based on the curated question and prompt query context.")
sql_agent_graph.add_node(is_safe_sql, name="Is Safe SQL?", description="Determines whether the generated SQL query is safe to execute or not.")
sql_agent_graph.add_node(canceled_sql, name="Canceled SQL Execution", description="Cancels the SQL query execution if it is deemed unsafe.")
sql_agent_graph.add_node(execute_sql, name="Execute SQL Query", description="Executes the generated SQL query against the database.")
sql_agent_graph.add_node(represent_final_answer, name="Represent Final Answer", description="Generates the final answer based on the SQL query execution result and the curated question.")

# edges
sql_agent_graph.add_edge(START, "curate_question")
sql_agent_graph.add_edge("curate_question", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")

# conditional edges based on the safety of the SQL query
def is_safe_condition(state: AgentSchema) -> str:
    is_safe = state.is_safe

    if is_safe.lower() == "yes":
        return "execute_sql"
    else:
        return "canceled_sql"

sql_agent_graph.add_conditional_edges("is_safe_sql", is_safe_condition, ["execute_sql", "canceled_sql"])
# execute
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)
# canceled
sql_agent_graph.add_edge("canceled_sql", END)


if __name__ == "__main__":
    # compile the graph to check for any issues
    sql_analyst = sql_agent_graph.compile()

    # show diagram of the graph
    img = Image(sql_analyst.get_graph().draw_mermaid_png())

    with open("sql_analyst_graph.png", "wb") as f:
        f.write(img.data)

    input_schema = {
        "messages": [],
        "user_input": "What are the different types of payment methods we have in our database?",
        "curated_question": "",
        "prompt_query_context": "",
        "is_safe": "No",
        "comments": "",
        "generated_sql_query": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    # execute graph 
    sql_analyst_response = sql_analyst.invoke(input_schema)
    print("Final Answer: ", sql_analyst_response['final_answer'])

