import json
import argparse
from tqdm import tqdm 
import os
import openai
import shutil
import re
from prompts import clarification_user_01, clarification_user_02, clarification_AI_01
from utils import is_penultimate_directory, list2prompt, get_llm_response


def collect_data_dirs(target_path, condition="clarification"):
    data_dirs = []
    for root, _, _ in os.walk(target_path): 
        if is_penultimate_directory(root): 
            if  "atp_tennis" in root or "credit_card_risk" in root or "fast_food" in root or "laptop_price" in root or "melb_housing" in root:
                # condition
                if condition in root:
                    data_dirs.append(root)

    return data_dirs

def llm_ask_for_clarification(all_roots, output_path, llm_engine, role='assistant'):
    question_all = {}
    count_save = 0
    engine_name = llm_engine.replace("-", "_")
    plotqa_prefix = "01_ask_for_clarification_"
    save_dir_01 = os.path.join(output_path, 'normal/' + plotqa_prefix + engine_name + '.json')
    try:
        with open(save_dir_01, 'r') as f:
            question_all = json.load(f)
    except:
        pass

    for root in tqdm(all_roots, desc="Processing clarification step 01"):
        if root in question_all and question_all[root] != "Failed!":
            continue

        count_save += 1
        try:
            with open(os.path.join(root, 'src/prompt_code_hist_origin.json'), 'r') as f_w:  
                list_prompt = json.load(f_w)

            with open(os.path.join(root, 'src/prompt_code_hist_origin.txt'), 'r') as f_w:  
                tmp = f_w.read()

        except FileNotFoundError:
            src_file = os.path.join(root, 'src/prompt_code_hist.json')
            dst_file = os.path.join(root, 'src/prompt_code_hist_origin.json')
            shutil.copy(src_file, dst_file)

            src_file = os.path.join(root, 'src/prompt_code_hist.txt')
            dst_file = os.path.join(root, 'src/prompt_code_hist_origin.txt')
            shutil.copy(src_file, dst_file)

        with open(os.path.join(root, 'src/prompt_code_hist_origin.json'), 'r') as f_w:  
            list_prompt = json.load(f_w)

        cut_idx = list_prompt[-2]["content"].find("My template of code snippet is:")
        list_prompt[-2]["content"] = list_prompt[-2]["content"][:cut_idx] + "\n" +  clarification_user_01
        list_prompt[-1]["content"] ="<question>"

        prompt_input_text = list2prompt(list_prompt)
        prompt_input_text = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', prompt_input_text)
        message_obj = [{"role": role, "content": prompt_input_text}]

        llm_response = get_llm_response(message_obj, engine=llm_engine)
        message_llm = llm_response["choices"][0]["message"]["content"]

        question_all[root] = message_llm

        if count_save % 2 == 0:
            with open(save_dir_01, "w") as f_out:
                json.dump(question_all, f_out, indent = 4)

    with open(save_dir_01, 'w') as f:
        json.dump(question_all, f, indent=4)

    return save_dir_01

def user_simulator(all_roots, output_path, llm_engine, save_dir_question, role='assistant'):
    with open(save_dir_question, 'r') as f:
        question_all = json.load(f)

    answer_simu_dict = {}
    count_save = 0
    engine_name = llm_engine.replace("-", "_")
    plotqa_prefix = "02_user_simulator_"
    save_dir = os.path.join(output_path, 'normal/' + plotqa_prefix + engine_name + '.json')
    try:
        with open(save_dir, 'r') as f:
            answer_simu_dict = json.load(f)
    except:
        pass

    for root in tqdm(all_roots, desc="Processing answer step 02"):
        if root in answer_simu_dict and answer_simu_dict[root] != "Failed!":
            continue

        count_save += 1
        with open(os.path.join(root, 'ref_code.py'), 'r') as f_w: 
            ref_code = f_w.read() 

        with open(os.path.join(root, 'src/prompt_code_hist_origin.json'), 'r') as f_w:  
            list_prompt = json.load(f_w)
            cut_idx = list_prompt[-2]["content"].find("My template of code snippet is:")
            list_prompt[-2]["content"] = list_prompt[-2]["content"][:cut_idx] + "\n" +  clarification_user_01
            cut_idx = question_all[root].find("</question>")
            if cut_idx != -1:
                list_prompt[-1]["content"] = question_all[root][:cut_idx].replace("<question>", "")
            else:
                list_prompt[-1]["content"] = question_all[root].replace("<question>", "")
            prompt_add = {"role": "user"}
            prompt_add["content"] = clarification_AI_01.format(ref_code=ref_code)
            list_prompt.append(prompt_add)
            prompt_add = {"role": "assistant"}
            prompt_add["content"] = "<answer>"
            list_prompt.append(prompt_add)
            
        prompt_input_text = list2prompt(list_prompt)
        prompt_input_text = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', prompt_input_text)
        message_obj = [{"role": role, "content": prompt_input_text}]

        llm_response = get_llm_response(message_obj, engine=llm_engine)
        message_llm = llm_response["choices"][0]["message"]["content"]

        answer_simu_dict[root] = message_llm

        if count_save % 2 == 0:
            with open(save_dir, "w") as f_out:
                json.dump(answer_simu_dict, f_out, indent = 4)

    with open(save_dir, 'w') as f:
        json.dump(answer_simu_dict, f, indent=4)

    return save_dir


def save_prompts(all_roots, save_dir_question, save_dir_answer):
    
    with open(save_dir_answer, 'r') as f:
        answer_simu_dict = json.load(f)

    with open(save_dir_question, 'r') as f:
        question_all = json.load(f)

    for root in tqdm(all_roots, desc="Processing save prompts:"):
        with open(os.path.join(root, 'src/prompt_code_hist_origin.json'), 'r') as f_w:  
            list_prompt = json.load(f_w)
            
            list_prompt[-2]["content"] = list_prompt[-2]["content"]
            cut_idx = question_all[root].find("</question>")
            if cut_idx != -1:
                list_prompt[-1]["content"] = question_all[root][:cut_idx].replace("<question>", "")
            else:
                list_prompt[-1]["content"] = question_all[root].replace("<question>", "")
            prompt_add = {"role": "user"}
            cut_idx = answer_simu_dict[root].find("</answer>")
            if cut_idx != -1:
                prompt_add["content"] = clarification_user_02.format(clarification=answer_simu_dict[root][:cut_idx].replace("<answer>", ""))
            else:
                prompt_add["content"] = clarification_user_02.format(clarification=answer_simu_dict[root].replace("<answer>", ""))
            list_prompt.append(prompt_add)
            prompt_add = {"role": "assistant"}
            prompt_add["content"] = ""
            list_prompt.append(prompt_add)

        prompt_hist = list2prompt(list_prompt)
        prompt_hist = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', prompt_hist)
                
        with open(os.path.join(root, 'src/prompt_code_hist.json'), 'w') as f_w:  
            json.dump(list_prompt, f_w, indent=4)

        with open(os.path.join(root, 'src/prompt_code_hist.txt'), 'w') as f_w:  
            f_w.write(prompt_hist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the tapilot data directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4-32k", required=True, help="Model name of chosen LLM.")
    parser.add_argument("--user_simulator_model_name", type=str, default="gpt-4-32k", required=True, help="Model name of chosen LLM for user simulator.")
    parser.add_argument("--api_key", type=str, required=True, help="Your openai api_key.")
    args = parser.parse_args()
    
    ############ Hyper Paras ############
    role = 'assistant'
    openai.api_key = args.api_key
    llm_engine = args.llm_model_name
    engine_name = llm_engine.replace("-", "_")
    ####################################

    data_dirs = collect_data_dirs(args.data_path)
    save_dir_question = llm_ask_for_clarification(data_dirs, args.output_path, llm_engine)
    save_dir_answer = user_simulator(data_dirs, args.output_path, args.user_simulator_model_name, save_dir_question)
    save_prompts(data_dirs, save_dir_question, save_dir_answer)
