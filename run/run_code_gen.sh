#!bin/bash

# Openai API key
api_key="YOUR-OPENAI-API"
# Your path to tapilot dialogue data
data_path="YOUR-PATH-OF-TAPILOT-DATA" # For example: /YOUR-DIR-PREFIX/data/interaction_data
# Your path to your output files
output_path="PATH-OF-OUTPUT-FILE" # For example: /YOUR-DIR-PREFIX/output
# The path to decision_company.json
private_lib_path="PATH-OF-PRIVATE-LIB-JSON-FILE" # For example: /YOUR-DIR-PREFIX/data/interaction_data/decision_company.json
# Your LLM model name to choose
llm_model_name="YOUR-LLM" # For example: gpt-4-32k
# Model settings choose from: ["base", "agent", "inter_agent"]
model_version="MODEL-VERSION" # For example: base
# Action type settings choose from: ["None", "correction", "clarification"]
action_type="None" # For example: correction

python3 ./methods/tapilot_code_gen.py --data_path $data_path --output_path $output_path --private_lib_path $private_lib_path --llm_model_name $llm_model_name --model_version $model_version --api_key $api_key --action_type $action_type
