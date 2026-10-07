"""
Central place to choose which LLM a task uses.

Not every step of the agent needs the most powerful (and most expensive) model.
Rewording a question is easy; writing correct SQL is harder. So each node asks
for a "level" and this function maps it to a model:

    "low"      -> cheap, fast model   (curate the question, write the final answer)
    "medium"   -> stronger model      (generate SQL, judge if the SQL is safe)
    "advanced" -> strongest model     (not used yet)

Keeping this in one function means you can swap models for the whole project
by editing a single file.

Learning notes:
- temperature=0 makes the output as deterministic as possible. Good for SQL and
  classification, where you want the same answer every time, not creativity.
- max_tokens caps the length of the reply (and so the cost of each call).
- ChatOpenAI reads OPENAI_API_KEY from the environment, which load_dotenv()
  fills in from the .env file.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def pick_llm(level: str):
    """
    Return a LangChain chat model for the given cost/quality level.

    Args:
        level (str): "low", "medium" or "advanced" (case-insensitive).

    Returns:
        ChatOpenAI: a chat model. Call .invoke("prompt") on it; the reply text
        is in the .content attribute of the returned message.

    Raises:
        ValueError: if level is not one of the three options.
        (Note: the error message still lists the old names basic/intermediate/advanced.)
    """

    if level.lower() == "low":
        llm = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0, max_tokens=1500)
    elif level.lower() == "medium":
        llm = ChatOpenAI(model_name="gpt-5.6-terra", temperature=0, max_tokens=1500)
    elif level.lower() == "advanced":
        llm = ChatOpenAI(model_name="gpt-5.6-sol", temperature=0, max_tokens=1500)
    else:
        raise ValueError("Invalid level provided. Choose from 'basic', 'intermediate', or 'advanced'.")

    return llm

# Quick smoke test: `uv run utils/llm_pick.py` should print a reply from the model.
if __name__ == "__main__":
    llm = pick_llm("low")
    print(llm.invoke("What is the capital of France?"))
