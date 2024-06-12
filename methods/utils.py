import time
import os
import openai
import sys
from io import StringIO
import contextlib
import pandas as pd
import ast
import matplotlib.pyplot as plt


def get_llm_response(messages, engine="gpt-4-32k", temperature=0, max_tokens=2000, top_p=1, frequency_penalty=0, presence_penalty=0, timeout=10, stop=None):  
    MAX_API_RETRY = 10
    for i in range(MAX_API_RETRY):
        time.sleep(2) # try to avoid reaching rate limit
        try:
            response = openai.ChatCompletion.create(  
                engine=engine,  
                messages=messages,  
                temperature=temperature,  
                max_tokens=max_tokens,  
                top_p=top_p,  
                frequency_penalty=frequency_penalty,  
                presence_penalty=presence_penalty,  
                timeout=timeout,  
                stop=stop  
            )  
            break
        except Exception as e:
            print(e)
            time.sleep(10)
            response ='None'
        print('retry: ', i + 1)    

    return response


def is_penultimate_directory(path):  
    """  
    This function checks if the given path is a penultimate directory (i.e., its subdirectories do not contain any other directories).  
  
    :param path: Path to the directory to check  
    :return: True if the directory is penultimate, False otherwise  
    """  
    # List all subdirectories in the given path  
    subdirectories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]  
  
    # If there are no subdirectories, the given path is not a penultimate directory  
    if not subdirectories:  
        return False  
  
    # Check each subdirectory to see if it contains any other directories  
    for subdir in subdirectories:  
        subdir_path = os.path.join(path, subdir)  
        subdir_subdirectories = [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))]  
  
        # If a subdirectory contains other directories, the given path is not a penultimate directory  
        if subdir_subdirectories:  
            return False  
  
    # If none of the subdirectories contain other directories, the given path is a penultimate directory  
    return True 


@contextlib.contextmanager  
def capture_output():  
    new_out, new_err = StringIO(), StringIO()  
    old_out, old_err = sys.stdout, sys.stderr  
    try:  
        sys.stdout, sys.stderr = new_out, new_err  
        yield sys.stdout, sys.stderr  
    finally:  
        sys.stdout, sys.stderr = old_out, old_err 


def process_python_output(stdout):
    # Process the captured output
    processed_output = ""
    for line in stdout.splitlines():
        try:
            # Try to evaluate each line as a literal (e.g., dict, list, etc.)
            # This is safe because ast.literal_eval can only evaluate literals
            value = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            # If it's not a literal, just print the line as is
            processed_output += line + "\n"
        else:
            # Special formatting for dicts and DataFrames
            if isinstance(value, dict):
                if len(value) > 15:
                    processed_output += str({k: value[k] for k in list(value)[:3]}) + "...\n"
                else:
                    processed_output += str(value) + "\n"
            elif isinstance(value, pd.DataFrame):
                processed_output += str(value.head(3)) + "\n... ...\n"
            elif isinstance(value, plt.Figure):
                processed_output += "Here is a plot output." + "\n"
            else:
                processed_output += str(value) + "\n"

    return processed_output


def format_code(code_gen, root_base):
    for line in code_gen.split("\n"):
        if "pd.read_csv(" in line:
            if "atp_tennis" in line:
                new_line = line.replace("atp_tennis.csv", os.path.join(root_base, "atp_tennis.csv"))
            elif "credit_customers" in line:
                new_line = line.replace("credit_customers.csv", os.path.join(root_base, "credit_customers.csv"))
            elif "fastfood" in line:
                new_line = line.replace("fastfood.csv", os.path.join(root_base, "fastfood.csv"))
            elif "laptops_price" in line:
                new_line = line.replace("laptops_price.csv", os.path.join(root_base, "laptops_price.csv"))
            elif "melb_data" in line:
                new_line = line.replace("melb_data.csv", os.path.join(root_base, "melb_data.csv"))
            else:
                new_line = line
            code_gen = code_gen.replace(line, new_line)
        if "pickle.dump(" in line:
            code_gen = code_gen.replace(line, "")

    return code_gen


def remove_pickle_code(hist_code):
    for line in hist_code.split("\n"):
        if "pickle.dump(" in line or "plt.savefig(" in line:
            hist_code = hist_code.replace(line, "")

        if "print(" in line:
            indent = line.rfind("print(")
            if indent == 0:
                hist_code = hist_code.replace(line, "")
            else:
                hist_code = hist_code.replace(line, line[:indent]+"PLACE_HOLDER = True")

    return hist_code


def find_last_template(text, sym):
    last_triple_quotes_index = text.rfind(sym)  
    extracted_text = text[last_triple_quotes_index:] if last_triple_quotes_index != -1 else ""  

    return extracted_text


def find_template_before(text, sym):
    last_triple_quotes_index = text.rfind(sym)  
    extracted_text = text[:last_triple_quotes_index] if last_triple_quotes_index != -1 else ""  

    return extracted_text


def list2prompt(list_p):
    prompt = ""
    for dict in list_p:
        if dict["role"] == "system":
            prompt = prompt + dict["content"]
        if dict["role"] == "user":
            prompt = prompt + "\n\n" + "[USER (data scientist)]: " + dict["content"]
        if dict["role"] == "assistant":
            prompt = prompt + "\n\n" + "[YOU (AI assistant)]: " + dict["content"]

    return prompt