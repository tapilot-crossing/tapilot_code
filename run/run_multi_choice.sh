#!bin/bash

# Openai API key
api_key="YOUR-OPENAI-API"
# Your path to tapilot dialogue data
data_path="YOUR-PATH-OF-TAPILOT-DATA" # For example: /YOUR-DIR-PREFIX/data/interaction_data
# Your path to your output files
output_path="PATH-OF-OUTPUT-FILE" # For example: /YOUR-DIR-PREFIX/output/normal
# The path to decision_company.json
private_lib_path="PATH-OF-PRIVATE-LIB-JSON-FILE" # For example: /YOUR-DIR-PREFIX/data/interaction_data/decision_company.json
# Your LLM model name to choose
llm_model_name="YOUR-LLM" # For example: gpt-4-32k
# Model settings choose from: ["base", "agent", "inter_agent"]
model_version="MODEL-VERSION" # For example: base
# The action type you want to take. There are: "_analysis"; "_una"; "_bg"
data_select="YOUR-ACTION"
# The max turn allowed in ReAct prompt ("agent" & "inter_agent" setting)
max_turns=5

python3 ./methods/tapilot_multi_choice.py --data_path $data_path --output_path $output_path --private_lib_path $private_lib_path --llm_model_name $llm_model_name --model_version $model_version --data_select $data_select --max_turns $max_turns --api_key $api_key
