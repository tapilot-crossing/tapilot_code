import json
import os
from tapilot import tapilot_agent
import openai
import argparse
from utils import is_penultimate_directory


def collect_data_dirs(target_path, action=None):
    data_dirs = []
    for root, _, _ in os.walk(target_path): 
        if is_penultimate_directory(root):
            if  "atp_tennis" in root or "credit_card_risk" in root or "fast_food" in root or "laptop_price" in root or "melb_housing" in root:
                # condition
                if action != "None":
                    if action in root:
                        data_dirs.append(root)
                else:
                    if "/action" in root or "_action" in root:
                        continue
                    data_dirs.append(root)

    return data_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the tapilot data directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--private_lib_path", type=str, required=True, help="Path to the private library directory.")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4-32k", required=True, help="Model name of chosen LLM.")
    parser.add_argument("--model_version", type=str, default="base", choices=['base', 'agent', 'inter_agent'], required=True, help="Model version of chosen LLM.")
    parser.add_argument("--action_type", type=str, default="None", choices=['None', 'correction', 'clarification'], required=True, help="The action type, if applied.")
    parser.add_argument("--api_key", type=str, required=True, help="Your openai api_key.")
    args = parser.parse_args()
    
    ############ Hyper Paras ############
    role = 'assistant'
    openai.api_key = args.api_key
    llm_engine = args.llm_model_name
    engine_name = llm_engine.replace("-", "_")
    ####################################

    with open(args.private_lib_path, 'r') as f_r:  
        decision_company = json.load(f_r) 

    data_dirs = collect_data_dirs(args.data_path, args.action_type)
    tapilot_model = tapilot_agent(data_dirs, args.output_path, engine_name, args.model_version, decision_company, args.data_path)

    if args.model_version == "base" or args.model_version == "agent":
        tapilot_model.code_gen_base_agent()
    elif args.model_version == "inter_agent":
        tapilot_model.code_gen_inter_agent()
    else:
        print("Wrong model_version, please input again!")
        raise AssertionError