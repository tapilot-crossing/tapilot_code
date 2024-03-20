#!bin/bash

# Openai API key
api_key="YOUR-OPENAI-API"
# Your path to tapilot dialogue data
data_path="YOUR-PATH-OF-TAPILOT-DATA" # For example: /YOUR-DIR-PREFIX/data/interaction_data
# Your path to your output files
output_path="PATH-OF-OUTPUT-FILE" # For example: /YOUR-DIR-PREFIX/output/normal
# Your LLM model name to choose
llm_model_name="YOUR-LLM" # For example: gpt-4-32k
# Model settings choose from: ["base", "agent", "inter_agent"]
model_version="MODEL-VERSION" # For example: base

python3 ./methods/tapilot_plotqa.py --data_path $data_path --output_path $output_path --llm_model_name $llm_model_name --model_version $model_version --api_key $api_key
