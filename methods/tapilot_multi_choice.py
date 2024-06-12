import json
import os
from tapilot import tapilot_agent
import openai
import argparse
from utils import is_penultimate_directory


def collect_data_dirs(target_path, condition):
    data_dirs = []
    for root, _, _ in os.walk(target_path): 
        if is_penultimate_directory(root): 
            if  "atp_tennis" in root or "credit_card_risk" in root or "fast_food" in root or "laptop_price" in root or "melb_housing" in root:
                # condition
                if condition in root:
                    data_dirs.append(root)

    return data_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the tapilot data directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--private_lib_path", type=str, required=True, help="Path to the private library directory.")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4-32k", required=True, help="Model name of chosen LLM.")
    parser.add_argument("--model_version", type=str, default="base", choices=['base', 'agent', 'inter_agent'], required=True, help="Model version of chosen LLM.")
    parser.add_argument("--data_select", type=str, default="_analysis", choices=['_analysis', '_una', '_plotqa', '_bg'], required=True, help="Select what you want from multi-choice data.")
    parser.add_argument("--max_turns", type=int, default=5, required=True, help="Select the max number of interaction turns.")
    parser.add_argument("--api_key", type=str, required=True, help="Your openai api_key.")
    args = parser.parse_args()
    
    ############ Hyper Paras ############
    role = 'assistant'
    openai.api_key = args.api_key
    llm_engine = args.llm_model_name
    engine_name = llm_engine.replace("-", "_")
    ####################################

    # load private lib
    with open(args.private_lib_path, 'r') as f_r:  
        decision_company = json.load(f_r) 

    data_dirs = collect_data_dirs(args.data_path, args.data_select)
    tapilot_model = tapilot_agent(data_dirs, args.output_path, engine_name, args.model_version, decision_company, args.data_path)

    if args.model_version == "base":
        tapilot_model.multi_choice_base(args.data_select)
    elif args.model_version == "inter_agent" or args.model_version == "agent":
        tapilot_model.multi_choice_agents(args.data_select, args.max_turns)
    else:
        print("Wrong model_version, please input again!")
        raise AssertionError