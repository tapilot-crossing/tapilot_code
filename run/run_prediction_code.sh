#!/bin/bash  
  
# Your path to tapilot dialogue data  
DATA_PATH="TAPILOT-DATA-PATH" # For example: /YOUR-DIR-PREFIX/data/interaction_data
# The path to all resources files like all csv files and decision_company.json
CSV_PATH="CSV-FILES-PATH" # For example: /YOUR-DIR-PREFIX/data/resource
# The python code extracted from LLM response generated in postprocessing
FIL_NAME="YOUR-CODE-GEN-PYTHON-FILE-NAME" # For example: pred_code.py
  
success_count=0  
failed_count=0  
  
failed_files=()  
  
for dir in $(find $DATA_PATH -type d)  
do  
    if [ -f "$dir/$FIL_NAME" ]; then  
        original_dir=$(pwd)  

        cd "$dir" 
        rm -f pred_result/* 

        timeout 5m python3 "./$FIL_NAME" $CSV_PATH

        if [ $? -eq 0 ]; then  
            success_count=$((success_count + 1))  
        else  
            failed_count=$((failed_count + 1))  
            failed_files+=("$dir/$FIL_NAME")  
        fi  

        cd "$original_dir"  
    fi  
done  

echo "# successful ref_code_hist.py: $success_count"
echo "# Failed ref_code_hist.py: $failed_count"
