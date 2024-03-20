#!bin/bash

# The path of LLM response json file generated in "run_code_gen.sh"
llm_response_path="YOUR-PATH-OF-LLM-RESPONSE" # For example: /YOUR-DIR-PREFIX/output/normal/multi_choice_AIR_gpt_4_turbo.json
# The path of LLM response json file generated in "run_code_gen.sh" with "agent" setting to complement "inter_agent" setting due to no first turn performance in AIR
ref_response_path="YOUR-PATH-OF-REFERENCE-LLM-RESPONSE" # For example: /YOUR-DIR-PREFIX/output/normal/multi_choice_agent_gpt_4_turbo.json
# The action name you want to evaluate
action_name="YOUR-ACTION" # For example: analysis
# Your LLM model name to choose
llm_model_name="YOUR-LLM" # For example: gpt-4-32k
# Model settings choose from: ["base", "agent", "inter_agent"]
model_version="MODEL-VERSION" # For example: base

python3 ./eval/eval_multi_choice.py --llm_response_path $llm_response_path --ref_response_path $ref_response_path --action_name $action_name --llm_model_name $llm_model_name --model_version $model_version   