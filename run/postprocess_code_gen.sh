#!bin/bash

# The path of LLM response json file generated in "run_code_gen.sh"
llm_response_path="YOUR-PATH-OF-LLM-RESPONSE" # For example: /YOUR-DIR-PREFIX/output/normal/code_gen_AIR_gpt_4_turbo.json
# The path of private function extraction LLM response json file generated in "run_code_gen.sh" with "private" setting 
private_func_path="YOUR-PATH-OF-PRIVATE-FUNCTION" # For example: /YOUR-DIR-PREFIX/output/normal/private_func_agent_gpt_4_turbo.json
# The python code extracted from LLM response generated in postprocessing
code_seg_fn="PREDICTED-CODE-SEGMENT-FILE-NAME" # For example: pred_code_segment.py
# The python code extracted from LLM response generated in postprocessing
code_pred_fn="PREDICTED-CODE-FILE-NAME" # For example: pred_code.py
# The mode of model you want to process: choose from ["base", "agent", "inter_agent"]
MODE="PLEASE-CHOOSE-MODE" # For example: base


if [ $MODE == "base" ]; then
    python3 ./postprocessing/combine_code_gen_base.py --llm_response_path $llm_response_path --private_func_path $private_func_path --code_seg_fn $code_seg_fn --code_pred_fn $code_pred_fn 
fi

if [ $MODE == "agent" ]; then
    python3 ./postprocessing/combine_code_gen_COT.py --llm_response_path $llm_response_path --code_seg_fn $code_seg_fn --code_pred_fn $code_pred_fn 
fi

if [ $MODE == "inter_agent" ]; then
    python3 ./postprocessing/combine_code_gen_AIR.py --llm_response_path $llm_response_path --code_seg_fn $code_seg_fn --code_pred_fn $code_pred_fn 
fi
