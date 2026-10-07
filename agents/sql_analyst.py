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
    └── run.py

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
from models.schema import AgentSchema

# test if llm is working
#llm = pick_llm("low")
#print(llm.invoke("What is the capital of France?"))

# python functions are my nodes in the graph, and the edges are the calls between them. I can use this to build a graph of function calls and visualize it.

# ------------------------------------------------------- AI Agent code ----------------------------------------------------------------------------------

# 1st node
def curate_question(state: AgentSchema) -> AgentSchema: 
    # We hsould always return the entire state, so that the next node can use it, to maintain the state between nodes. 
    # This is important for the graph structure, as each node can modify the state and pass it along to the next node.
    """
    Curates a question based on the current state of the agent.

    Args:
        state (AgentSchema): The current state of the agent.

    Returns:
        str: The curated question.
    """
    
    user_question = state.user_input #this is a pydantic model object
    llm = pick_llm("low") #pick a low cost llm for this task, as it is not critical to have a high quality answer for this task.
    response = llm.invoke(f"Curate the following question: {user_question}")
    state.curated_question = response

    return state 

# 2nd node
def prompt_query_context(state: AgentSchema) -> AgentSchema:
    """
    Generates a detailed prompt with SQL DB context that will help the agent to generate SQL query.

    Args:
        state (AgentSchema): The current state of the agent.

    Returns:
        str: The detailed prompt with SQL DB context.
    """

    curated_question = state.curated_question 


    

    return state