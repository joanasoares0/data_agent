"""
SQL Analyst Agent: ask a question in plain English, get an answer from Postgres.

The agent is a LangGraph *state graph*: a set of Python functions (nodes)
connected by edges. A shared state object (AgentSchema) flows through the
graph, and each node reads from it and writes its results back.

Flow:

    START
      -> curate_question        (LLM, low)     rewrite the question clearly
      -> prompt_query_context   (no LLM)       build a prompt with the DB schema
      -> generate_sql           (LLM, medium)  write the SQL query
      -> is_safe_sql            (LLM, medium)  judge: is the query read-only?
           |-- "Yes" -> execute_sql             run it on Postgres
           |               -> represent_final_answer (LLM, low) explain the result
           |               -> END
           |-- "No"  -> canceled_sql            stop and explain why
                           -> END

Run it with:  uv run agents/sql_analyst.py
It also saves a picture of the graph to sql_analyst_graph.png.
"""

import os 
import sys

# Add the parent directory to the system path to allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# How the line above works
# ------------------------
# It lets a Python script import modules from the folder one level above the one
# where the script lives. Example:
#
#   project/
#   ├── utils.py
#   └── scripts/
#       └── run.py
#
# Read it from the inside out:
#   __file__                      path of the current script, e.g. project/scripts/run.py
#   os.path.dirname(__file__)     its folder: project/scripts
#   os.path.join(..., '..')       "go up one level": project/scripts/..
#   os.path.abspath(...)          clean full path: /home/you/project
#   sys.path.append(...)          add it to the folders Python searches on `import`
#
# Without it, `import utils` inside run.py fails, because Python only looks in
# scripts/ (and installed packages). With it, Python also looks in project/.
#
# Notes:
#   - It needs `import os` and `import sys` above.
#   - It works, but it's a workaround. Cleaner options: run the script as a module
#     from the project root (`python -m scripts.run`) or make the project an
#     installable package (`pip install -e .`). Fine for a small project; in a
#     larger codebase it makes imports harder to follow.

# --- Project imports ----------------------------------------------------------
from utils.llm_pick import pick_llm              # chooses the LLM by cost level
from models.schema import AgentSchema, JudgeSchema  # graph state + judge output shape
from utils.database import DatabaseUtil          # Postgres helper

# --- LangChain / LangGraph ------------------------------------------------------
# HumanMessage / AIMessage: chat message types, used to keep the conversation history.
from langchain_core.messages import AIMessage, HumanMessage
# StateGraph builds the graph; START and END are the special entry/exit nodes.
from langgraph.graph import StateGraph, START, END

# Image wraps the PNG bytes of the graph diagram (saved to a file below).
from IPython.display import display, Image

# test if llm is working
#llm = pick_llm("low")
#print(llm.invoke("What is the capital of France?"))

# Python functions are the nodes of the graph, and the edges define which node
# runs next. LangGraph can also draw the graph as a diagram.

# ======================================================================================
# NODES
# Every node has the same shape:
#   - input:  the current state (an AgentSchema object)
#   - output: a dict with ONLY the fields it changed
# LangGraph merges that dict into the state before calling the next node.
# ======================================================================================

# --------------------------------------------------------------------------------------
# Node 1: curate_question
# --------------------------------------------------------------------------------------
def curate_question(state: AgentSchema) -> dict:
    # Each node returns only the fields it changed, as a dict. LangGraph merges that into the state.
    # Don't return the whole state: `messages` uses the `add` reducer, so returning the full list
    # would append every existing message again (duplicates). Return just the new messages instead.
    """    
    Rewrite the user's question so it is clearer and more specific.

    A cheap model is enough here: it's a simple rewording task, and a cleaner
    question makes the SQL step more accurate.

    Args:
        state (AgentSchema): The current state of the agent.

    Returns:
        dict: {"curated_question": ..., "messages": [HumanMessage]}
    """    

    user_question = state.user_input # state is a Pydantic object, so fields are read with dot notation
    llm = pick_llm("low") #pick a low cost llm for this task, as it is not critical to have a high quality answer for this task.

    # .invoke() returns an AIMessage; .content is its text.
    response = llm.invoke(f"Curate the following question: {user_question}").content

    # the add reducer appends this new message to the conversation history
    return {"curated_question": response, "messages": [HumanMessage(content=response)]}

# --------------------------------------------------------------------------------------
# Node 2: prompt_query_context
# --------------------------------------------------------------------------------------
def prompt_query_context(state: AgentSchema) -> dict:
    """    
    Build the prompt for the SQL generator: instructions + question + DB schema.

    No LLM call here. The node reads the schema from Postgres (tables, columns,
    types and sample rows) and pastes it into the prompt. This is the "context"
    step: the LLM can only write correct SQL if it knows the real table and
    column names.

    Args:
        state (AgentSchema): The current state of the agent.

    Returns:
        dict: {"prompt_query_context": <full prompt text>}
    """    

    curated_question = state.curated_question 

    # Connection settings come from environment variables (loaded from .env),
    # so passwords are never written in the code.
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

    # Constructing the prompt query for the agent to generate the SQL query.
    # Prompt-writing tips used here:
    #   - give the model a role ("You are an SQL analyst agent")
    #   - set guardrails (LIMIT 10 by default)
    #   - say exactly what to output (only SQL, no explanation), because the
    #     reply is executed as-is
    #   - tell it the schema name, so it writes rideshare.payments, not payments
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


# --------------------------------------------------------------------------------------
# Node 3: generate_sql
# --------------------------------------------------------------------------------------
def generate_sql(state: AgentSchema) -> dict:
    """    
    Ask the LLM to write the SQL query from the prompt built in node 2.

    Uses a stronger ("medium") model because this is the step where quality
    matters most: a wrong JOIN or column name means a wrong answer.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        dict: {"generated_sql_query": <SQL text>}
    """    

    prompt = state.prompt_query_context

    llm = pick_llm("medium") #pick a medium cost llm for this task, as it is critical to have a high quality answer for this task.
    generated_sql_query = llm.invoke(prompt).content

    return {"generated_sql_query": generated_sql_query}


# --------------------------------------------------------------------------------------
# Node 4: is_safe_sql  (the "LLM as a judge" guardrail)
# --------------------------------------------------------------------------------------
def is_safe_sql(state: AgentSchema) -> dict:
    """    
    Ask a second LLM call to check that the SQL only READS data.

    The answer comes back as structured output (JudgeSchema), so is_safe is
    always exactly "Yes" or "No" and the router below can rely on it.

    Note: an LLM judge is a helpful extra check, not a hard guarantee. The
    strongest protection is a database user that only has read (SELECT) rights.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        dict: {"is_safe": "Yes" | "No", "comments": <judge's reasoning>}
    """    
    sql_query = state.generated_sql_query

    llm = pick_llm("medium") #pick a medium cost llm for this task, as it is critical to have a high quality answer for this task.

    # with_structured_output(JudgeSchema) makes the model fill in JudgeSchema's
    # fields instead of replying with free text.
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

# --------------------------------------------------------------------------------------
# Node 5a: canceled_sql  (the "No" branch)
# --------------------------------------------------------------------------------------
def canceled_sql(state: AgentSchema) -> dict:
    """    
    Stop without running the query, and tell the user why (using the judge's comments).

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        dict: {"final_answer": <cancel message>, "messages": [AIMessage]}
    """    

    comments = state.comments
    final_answer = f"SQL query execution has been canceled due to safety concerns. Comments from the judge: {comments}"

    return {"final_answer": final_answer, "messages": [AIMessage(content=final_answer)]}

# --------------------------------------------------------------------------------------
# Node 5b: execute_sql  (the "Yes" branch)
# --------------------------------------------------------------------------------------
def execute_sql(state: AgentSchema) -> dict:
    """    
    Run the generated SQL on Postgres.

    If the query fails, execute_query returns the error as text, so the next
    node can still explain what went wrong instead of the graph crashing.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        dict: {"sql_query_execution_result": <rows as text, or error message>}
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

# --------------------------------------------------------------------------------------
# Node 6: represent_final_answer
# --------------------------------------------------------------------------------------
def represent_final_answer(state: AgentSchema) -> dict:
    """    
    Turn the raw rows (e.g. "[('apple_pay',), ('paypal',)]") into a friendly answer.

    A cheap model is enough: the hard work (the SQL) is done, this is just
    presentation.

    Args:
        state (AgentSchema): The current state of the agent.
    Returns:
        dict: {"final_answer": <answer text>, "messages": [AIMessage]}
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


# ======================================================================================
# GRAPH
# Create the nodes and edges: each node is a function, each edge says what runs next.
# ======================================================================================

# The graph is typed with AgentSchema, so LangGraph knows the shape of the state
# (and validates the input against it).
sql_agent_graph = StateGraph(AgentSchema)

# --- Nodes ---
# The node's ID is the function name (e.g. "curate_question"); that's the name
# used in the edges below.
sql_agent_graph.add_node(curate_question, name="Curate Question", description="Curates the user's question to make it more clear and specific.")
sql_agent_graph.add_node(prompt_query_context, name="Prompt Query Context", description="Generates a detailed prompt with SQL DB context that will help the agent to generate SQL query.")
sql_agent_graph.add_node(generate_sql, name="Generate SQL Query", description="Generates an SQL query based on the curated question and prompt query context.")
sql_agent_graph.add_node(is_safe_sql, name="Is Safe SQL?", description="Determines whether the generated SQL query is safe to execute or not.")
sql_agent_graph.add_node(canceled_sql, name="Canceled SQL Execution", description="Cancels the SQL query execution if it is deemed unsafe.")
sql_agent_graph.add_node(execute_sql, name="Execute SQL Query", description="Executes the generated SQL query against the database.")
sql_agent_graph.add_node(represent_final_answer, name="Represent Final Answer", description="Generates the final answer based on the SQL query execution result and the curated question.")

# --- Edges (fixed path) ---
# add_edge(A, B) means "after A finishes, always run B".
sql_agent_graph.add_edge(START, "curate_question")
sql_agent_graph.add_edge("curate_question", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")

# --- Conditional edge (the branch) ---
# A router function reads the state and returns the NAME of the next node.
# It doesn't change the state; it only decides where to go.
def is_safe_condition(state: AgentSchema) -> str:
    is_safe = state.is_safe

    if is_safe.lower() == "yes":
        return "execute_sql"
    else:
        return "canceled_sql"

# The list gives all possible destinations, so LangGraph can validate and draw them.
sql_agent_graph.add_conditional_edges("is_safe_sql", is_safe_condition, ["execute_sql", "canceled_sql"])
# "Yes" path: execute -> explain -> end
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)
# "No" path: cancel -> end
sql_agent_graph.add_edge("canceled_sql", END)


# ======================================================================================
# RUN
# ======================================================================================
if __name__ == "__main__":
    # compile() checks the graph (e.g. every edge points to a real node) and
    # returns a runnable object with .invoke().
    sql_analyst = sql_agent_graph.compile()

    # Save a diagram of the graph (rendered with Mermaid) to a PNG file.
    img = Image(sql_analyst.get_graph().draw_mermaid_png())

    with open("sql_analyst_graph.png", "wb") as f:
        f.write(img.data)

    # Initial state. Every required AgentSchema field needs a starting value;
    # the nodes fill in the empty ones as the graph runs.
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

    # Execute the graph. invoke() runs it from START to END and returns the
    # final state as a dict.
    sql_analyst_response = sql_analyst.invoke(input_schema)
    print("Final Answer: ", sql_analyst_response['final_answer'])
