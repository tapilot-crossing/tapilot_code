import argparse
import os
import re
import json
import openai
from tqdm import tqdm  
import time
from utils import is_penultimate_directory, get_llm_response, process_python_output, capture_output, format_code, remove_pickle_code


def collect_data_dirs(target_path, condition="_plotqa"):
    data_dirs = []
    for root, _, _ in os.walk(target_path): 
        if is_penultimate_directory(root): 
            if  "atp_tennis" in root or "credit_card_risk" in root or "fast_food" in root or "laptop_price" in root or "melb_housing" in root:
                # condition
                if condition in root:
                    data_dirs.append(root)

    return data_dirs


def plotqa_base(all_roots, output_path, llm_engine, role='assistant'):
    llm_response_dict_02 = {}
    save_cnt = 0
    engine_name = llm_engine.replace("-", "_")
    plotqa_prefix = "PlotQA_baseline_"
    save_dir_01 = os.path.join(output_path, 'multi_choice/' + plotqa_prefix + engine_name + '.json')
    try:
        with open(save_dir_01, "r") as f_ou:
            llm_response_dict_02 = json.load(f_ou)
    except:
        pass

    for root in tqdm(all_roots, desc="Processing plotqa baseline"):
        if root in llm_response_dict_02 and llm_response_dict_02[root] != "Failed!":
            continue

        save_cnt += 1
        with open(os.path.join(root, "reference/prompt_code_hist.txt"), 'r') as f_j: 
            prompt_hist = f_j.read()
        
        idx_cut_1 = prompt_hist.find("You are a data scientist with an impressive array of skills")
        idx_cut_2 = prompt_hist.rfind("--- Filtered Dataframe ---")
        prompt_hist = prompt_hist[:idx_cut_1] + prompt_hist[idx_cut_2:].replace("--- Filtered Dataframe ---", "")

        idx_cut = prompt_hist.rfind("[YOU (AI assistant)]")
        prompt_hist = prompt_hist[:idx_cut] + "Please choose your answer for the above multi-choice question between <answer>...</answer>.\n\n" + prompt_hist[idx_cut:].strip() + " <answer>"
        prompt_hist = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', prompt_hist)

        message_obj = [{"role": role, "content": prompt_hist}]

        llm_response = get_llm_response(message_obj, engine=llm_engine)
        message_llm = llm_response["choices"][0]["message"]["content"]

        llm_response_dict_02[root] = message_llm

        if save_cnt % 2 == 0:
            with open(save_dir_01, "w") as f_out:
                json.dump(llm_response_dict_02, f_out, indent = 4)

    with open(save_dir_01, "w") as f_ou:
        json.dump(llm_response_dict_02, f_ou, indent = 4)


def plotqa_agent(all_roots, output_path, data_path, llm_engine, role='assistant'):
    llm_response_dict_01 = {}
    llm_response_dict = {}
    save_cnt = 0
    engine_name = llm_engine.replace("-", "_")
    plotqa_prefix = "PlotQA_agent_"
    save_dir_01 = os.path.join(output_path, 'multi_choice/01_step_' + plotqa_prefix + engine_name + '.json')
    save_dir = os.path.join(output_path, 'multi_choice/02_step_' + plotqa_prefix + engine_name + '.json')

    try:
        with open(save_dir, "r") as f_out:
            llm_response_dict = json.load(f_out)
    except:
        pass

    try:
        with open(save_dir_01, "r") as f_ou:
            llm_response_dict_01 = json.load(f_ou)
    except:
        pass

    for root in tqdm(all_roots, desc="Processing roots"):
        if root in llm_response_dict and llm_response_dict[root] != "Failed!":
            continue
        save_cnt += 1
        with open(os.path.join(root, "reference/ref_code_hist.py"), 'r') as f_r:  
            hist_code = f_r.read() 

        with open(os.path.join(root, "reference/prompt_code_hist.txt"), 'r') as f_j: 
            prompt_hist = f_j.read()
        
        prompt_hist = prompt_hist + " I will generate code which can assist me to answer the question between <code> MY-PYTHON-CODE </code> in this step. And I will use a 'print()' to print out my interested value at the end of my code.\n\n<code>"
        prompt_hist = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', prompt_hist)
        message_obj = [{"role": role, "content": prompt_hist}]

        llm_response = get_llm_response(message_obj, engine=llm_engine)
        message_llm = llm_response["choices"][0]["message"]["content"]

        llm_response_dict_01[root] = message_llm
        idx_1 = message_llm.find("<code>")
        idx_2 = message_llm.find("</code>")
        if idx_1 != -1 and idx_2 != -1:
            code_gen = message_llm[idx_1:idx_2].replace("<code>", "")
        else:
            code_gen = message_llm[:idx_2]

        code_gen = format_code(code_gen, data_path)
        hist_code = remove_pickle_code(hist_code)
        
        code_pred = hist_code + "\n" + code_gen
        code_pred = code_pred.replace("target_customer_segments = [1, 2]", "")
        code_pred = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', code_pred)

        with capture_output() as (out, err):
            try:
                exec(code_pred)  
            except Exception as e:
                print(f"An error occurred: {e}", file=err)

        stdout, stderr = out.getvalue(), err.getvalue()
        processed_output = process_python_output(stdout)
        
        if "An error occurred:" in stderr:  
            print("Standard Error:", root)  
            print(stderr) 
            processed_output = stderr

        trigger_sent = "[YOU (AI assistant)]:"
        idx_cut = prompt_hist.rfind(trigger_sent)
        if idx_cut != -1:
            prompt_hist = prompt_hist[:idx_cut]
        new_prompt = "--- Code Generated: ---\n'''" + code_gen + "'''\n\n--- Code Execution Result: ---\n" + processed_output + "\n\nBased on the code generated and its execution results for the first question of this turn, choose the most appropriate option to answer the latter question and directly provide the choice between <choice>...</choice>."
        prompt_hist = prompt_hist + new_prompt
        prompt_hist = prompt_hist + "\n\n" + "[YOU (AI assistant)]: <choice>"
        prompt_hist = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', prompt_hist)

        message_obj = [{"role": role, "content": prompt_hist}]
        llm_response = get_llm_response(message_obj, engine=llm_engine)
        message_llm = llm_response["choices"][0]["message"]["content"]

        llm_response_dict[root] = message_llm
        if save_cnt % 2 == 0:
            with open(save_dir, "w") as f_out:
                json.dump(llm_response_dict, f_out, indent = 4)

            with open(save_dir_01, "w") as f_ou:
                json.dump(llm_response_dict_01, f_ou, indent = 4)

    with open(save_dir, "w") as f_out:
        json.dump(llm_response_dict, f_out, indent = 4)

    with open(save_dir_01, "w") as f_ou:
        json.dump(llm_response_dict_01, f_ou, indent = 4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the tapilot data directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4-32k", required=True, help="Model name of chosen LLM.")
    parser.add_argument("--model_version", type=str, default="base", choices=['base', 'agent'], required=True, help="Model version of chosen LLM.")
    parser.add_argument("--api_key", type=str, required=True, help="Your openai api_key.")
    args = parser.parse_args()
    
    ############ Hyper Paras ############
    role = 'assistant'
    openai.api_key = args.api_key
    llm_engine = args.llm_model_name
    engine_name = llm_engine.replace("-", "_")
    ####################################

    data_dirs = collect_data_dirs(args.data_path)
    if args.model_version == "base":
        plotqa_base(data_dirs, args.output_path, llm_engine)
    elif args.model_version == "agent" or args.model_version == "inter_agent":
        plotqa_agent(data_dirs, args.output_path, args.data_path, llm_engine)
    else:
        print("Wrong model_version!")
        raise AssertionError
