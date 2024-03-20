#!bin/bash

# Your path to tapilot dialogue data
data_path="YOUR-PATH-OF-TAPILOT-DATA" # For example: /YOUR-DIR-PREFIX/data/interaction_data
# Your LLM model name to choose
llm_model_name="YOUR-LLM" # For example: gpt-4-32k

python3 ./eval/eval_one_click.py --data_path $data_path --llm_model_name $llm_model_name
