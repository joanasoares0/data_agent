from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def pick_llm(level: str):
    """
    Function to pick the appropriate LLM based on the given level.
    
    Args:
        level (str): The level of the LLM to be picked. Can be "basic", "intermediate", or "advanced".
        
    Returns:
        str: The name of the selected LLM.
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

# test if llm is working
if __name__ == "__main__":
    llm = pick_llm("low")
    print(llm.invoke("What is the capital of France?"))