#!/bin/bash  

# Your path to tapilot dialogue data  
DATA_PATH="TAPILOT-DATA-PATH" # For example: /YOUR-DIR-PREFIX/data/interaction_data
# The path to all resources files like all csv files and decision_company.json
csv_path="CSV-FILES-PATH" # For example: /YOUR-DIR-PREFIX/data/src

success_count=0  
failed_count=0
failed_files=()  
  
for dir in $(find $DATA_PATH -type d)  
do  
    if [ -f "$dir/ref_code_all.py" ]; then  
        original_dir=$(pwd)  
  
        cd "$dir" 
        cd ".."
  
        python3 "src/ref_code_all.py"  $csv_path
  
        if [ $? -eq 0 ]; then  
            success_count=$((success_count + 1))  
        else  
            failed_count=$((failed_count + 1))  
            failed_files+=("$dir/ref_code_all.py")  
        fi  
  
        cd "$original_dir"  
    fi  
done  
  
echo "# successful ref_code_all.py: $success_count"
echo "# Failed ref_code_all.py: $failed_count"
