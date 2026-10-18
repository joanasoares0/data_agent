"""
ETL helpers used by the ETL analyst agent (agents/etl_analyst.py).

The agent's tools are thin wrappers around these methods. Keeping the real
work here (plain Python, no LLM) means each step can be tested on its own.

Methods:

1. extract_load(url, destination, format)
   Extract + Load: call a web API, turn the JSON into a table (DataFrame),
   and save it as csv / json / parquet.

2. transform_load_context(file_path)
   Read a data file and return its first 3 rows as text. That text goes into
   the LLM prompt, so the model knows the columns before writing pandas code.

3. execute_code(code)
   Run a piece of Python code (written by the LLM) with exec().

Learning notes:
- requests is the standard library for calling web APIs (HTTP).
- pandas is the standard library for tables in Python. A DataFrame is a table
  with named columns, like a sheet in Excel.
- JSON from an API is often nested ({"results": [{"name": ..., "url": ...}]}).
  pd.json_normalize flattens it into rows and columns.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # read .env into os.environ (used by PATH_DATA_EXTRACTED in the test below)

class ETLTools:
    # No settings to keep yet, so __init__ does nothing. The class just groups
    # the related functions together.
    def __init__(self):
        pass

    # ----------------------------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------------------------
    def resolve_path(self, path: str) -> str:
        """
        Turn a path relative to the project root (e.g. "data/extracted") into a
        full path. Full paths are returned unchanged.

        Without this, "data/extracted" would mean "data/extracted inside the
        folder you ran the script from", which changes from run to run.
        """
        # project root = the folder above utils/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.join(project_root, path)  # join() keeps `path` as-is if it's already absolute

    # ----------------------------------------------------------------------------------
    # Extract + Load
    # ----------------------------------------------------------------------------------
    def extract_load(self, url: str, destination: str, format: str = "csv") -> str:
        """
        This tools extracts data from the API(url) and loads it into the desired location(destination)
        Args:
            url (str): The API endpoint to extract data from.
            destination (str): The destination where the extracted data will be loaded.
        Returns:
            str: A message indicating the success of the operation.
        """

        # Resolve paths relative to the project root,
        # e.g. destination "data/extracted" -> /.../data_agent/data/extracted,
        # so the file lands in the same place whatever folder you run from.
        output_folder = self.resolve_path(destination)

        try:
            # 1) Extract: call the API.
            response = requests.get(url)
            # raise_for_status() turns an HTTP error (404, 500, ...) into an exception,
            # so it's handled by the `except` below instead of being ignored.
            response.raise_for_status()
            data  = response.json()  # the response body as Python dicts/lists

            # 2) Prepare the output: file name + make sure the folder exists.
            # exist_ok=True: don't fail if the folder is already there.
            filename = os.path.join(output_folder, f"extracted_data.{format}")
            os.makedirs(output_folder, exist_ok=True)

            # 3) Transform the JSON into a table.
            # Many APIs (like PokeAPI) put the rows under a "results" key; others
            # return a list of rows, or a single object. Handle all three.
            if isinstance(data, dict) and "results" in data:
                rows = data["results"]
            else:
                rows = data
            # Normalize the JSON data and save it in the specified format
            df = pd.json_normalize(rows)

            # 4) Load: write the file in the requested format.
            # index=False: don't write pandas' row numbers as an extra column.
            # lines=True (json): one JSON object per line ("JSON Lines").
            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported format: {format}"

            # Errors and successes are both returned as text: the LLM reads this
            # message and tells the user what happened.
            return f"Data successfully extracted and saved to {filename}"

        except requests.exceptions.RequestException as e:
            # Network problems, HTTP errors (404, 500, ...), or a reply that isn't JSON.
            return f"Failed to extract data: {e}"
        except Exception as e:
            # Anything else (unexpected JSON shape, can't write the file, ...):
            # report it as text instead of crashing the whole graph.
            return f"Failed to save the extracted data: {e}"

    # ----------------------------------------------------------------------------------
    # Context for the transform step
    # ----------------------------------------------------------------------------------
    def transform_load_context(self, file_path:str):
        """
        This tool transforms the data from the specified file and loads it into the
        desired location (output_folder).

        Args:
            file_path (str): The path to the file containing the data to be transformed.
            output_folder (str): The folder where the transformed data will be saved.
            output_format (str): The format in which to save the transformed data (csv, json, parquet).
        Returns:
            str: A message indicating the success or failure of the operation.
        """
        # What it really does: read the file and return its first 3 rows as text
        # (the docstring above describes the whole tool, not this method).

        file_path = self.resolve_path(file_path)

        # splitext("data/x.csv") -> ("data/x", ".csv"); lower() so ".CSV" also works.
        file_extension = os.path.splitext(file_path)[1].lower()

        # Pick the right pandas reader for the file type.
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".json":
            df = pd.read_json(file_path, lines=True)
        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_extension}"

        # head(3) = first 3 rows; str() turns the table into readable text, e.g.
        #          name                                       url
        #   0   bulbasaur  https://pokeapi.co/api/v2/pokemon/1/
        top_3_rows = str(df.head(3))

        return top_3_rows

    # ----------------------------------------------------------------------------------
    # Run LLM-written code
    # ----------------------------------------------------------------------------------
    def execute_code(self,code:str):
        """
        This tool executes the provided code and returns the output.

        Args:
            code (str): The code to be executed.
        Returns:
            str: The output of the executed code or an error message if execution fails.
        """
        # exec(code) runs a string as Python code, inside THIS process.
        # ⚠ The code comes from an LLM, so it can do anything this program can:
        #   read .env, delete files, call the internet. OK for learning on your
        #   own machine; in production run it in a sandbox (see README, note 10).
        # Note: it returns a fixed message, not what the code printed or produced.

        try:
            exec(code)
            return "Code executed successfully."

        except Exception as e:
            # A bug in the generated code becomes a message, not a crash.
            return f"Failed to execute code: {e}"


# Quick manual test: `uv run utils/etl_tools.py`
# Needs PATH_DATA_EXTRACTED in .env (a folder path ending with "/").
if __name__ == "__main__":
    etl_tools = ETLTools()
    url = "https://pokeapi.co/api/v2/pokemon?limit=10"
    destination = "data/extracted"
    path = os.environ['PATH_DATA_EXTRACTED'] + "extracted_data.csv"
    #result = etl_tools.extract_load(url, destination, format="csv")
    result = etl_tools.transform_load_context(path)
    print(result)
