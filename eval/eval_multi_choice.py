import json
import os
import re
import argparse

# Function to parse the reference_answer.txt content and extract options and correct answer  
def parse_reference_answer_content(content):  
    options_match = re.search(r'"options":\s*\[(.*?)\]', content, re.DOTALL)  
    correct_answer_match = re.search(r'"correct_answer":\s*"([^"]+)"', content)  
    correct_answer_match_1 = re.search(r'"correct_answer":\s*([^"]+)', content)  
    if correct_answer_match:  
        options = []
        correct_answer = correct_answer_match.group(1).strip()  
        return options, correct_answer  
    elif correct_answer_match_1:  
        options = []
        correct_answer = correct_answer_match_1.group(1).strip()  
        return options, correct_answer  
    else:
        return [], ''  
  
# Function to extract and clean the provided answer for situation 1 and 2  
def extract_answer(text_with_answer):  
    # Handle situation 1: Only "</choice>" exists  
    if '</choice>' in text_with_answer and '\nAnswer:' not in text_with_answer:  
        choice_match = re.search(r'\b([A-J])\.?\b[^<]*</choice>', text_with_answer)  
        if choice_match:  
            return choice_match.group(1).upper()  # Extract the letter and convert to uppercase  
  
    # Handle situation 2: Only "\nAnswer:" exists  
    if '\nAnswer:' in text_with_answer and '</choice>' not in text_with_answer:  
        answer_match = re.search(r'\nAnswer: \b([A-J])\.?\b', text_with_answer)  
        if answer_match:  
            return answer_match.group(1).upper()  # Extract the letter and convert to uppercase  
        
    if '\nAnswer:' not in text_with_answer and '</choice>' not in text_with_answer: 
        choice_match = re.search(r'([A-J])\.?\b[^<]*', text_with_answer)  
        if choice_match:  
            return choice_match.group(1).upper()
  
    return ''  # Return empty string if no match found  

def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0

def eval_dict_initialize(path, setting, model_use, action_name):
    eval_jaon_list = []
    try:
        with open(os.path.join(path,"eval_stats.json"), "r") as f_eval:
            eval_jaon_list = json.load(f_eval)
    except:
        pass

    save_flg = True
    change_idx = -1
    if eval_jaon_list:
        for i in range(len(eval_jaon_list)):
            dict_i = eval_jaon_list[i]
            if dict_i["setting"] == setting and dict_i["model"] == model_use:
                save_flg = False
                change_idx = i
                break

    stats_dict = {"ex": 0, "exr": None, "class": [], "is_code": True, "valid": True, "model": model_use, "setting": setting}

    if is_folder_empty(os.path.join(path,"pred_result")):
        stats_dict["valid"] = False
    if "analysis" in path or "conclude" in path or "una_" in path or "bg_" in path or "plotqa" in path:
        stats_dict["is_code"] = False

    if "short" in path:
        stats_dict["class"].append("short")
    else:
        stats_dict["class"].append("long")

    if "private" in path:
        stats_dict["class"].append("private")
    else:
        stats_dict["class"].append("normal")

    if action_name in path:
        stats_dict["class"].append(action_name)

    return eval_jaon_list, stats_dict, save_flg, change_idx


def eval_main(json_file_path, setting, model_use, action_name):
    # Initialize counters  
    correct_count = 0  
    incorrect_count = 0  
    
    # Load the JSON file with answers  
    with open(json_file_path, 'r', encoding='utf-8') as f:  
        data = json.load(f)  

    total_questions = 0
    # Iterate over each entry in the JSON data  
    for path, text_with_answer in data.items():  
        eval_jaon_list, stats_dict, save_flg, change_idx = eval_dict_initialize(path, setting, model_use, action_name)
        total_questions += 1
        # Extract the answer from the text using the new function  
        answer = extract_answer(text_with_answer)  
        if answer == '':
            incorrect_count += 1
            stats_dict["ex"] = 0
            print("--------------------------------- NOT our template!!!!! ---------------------------------")
            print(path)
            print("---------------------------------------------------------------------------------------- ")

        else:
            # path = path.replace("/home/v-jinyangli/Tapilot/Tapilot_data/", "/Users/jinyangli/Desktop/Tapilot_data/")
            # Construct the path to the reference_answer.txt file  
            reference_answer_file_path = os.path.join(path, 'reference_answer.txt')  

            # Check if the reference_answer.txt file exists  
            if os.path.exists(reference_answer_file_path):  
                # Load the content of the reference_answer.txt file  
                with open(reference_answer_file_path, 'r', encoding='utf-8') as f:  
                    reference_answer_content = f.read()  

                # Parse the content to get the options and correct answer  
                options, correct_answer = parse_reference_answer_content(reference_answer_content)  

                # Check if the provided answer matches the correct answer directly  
                if answer and answer == correct_answer:  
                    correct_count += 1  
                    stats_dict["ex"] = 1
                    print(f'**********The answer for path "{path}" is correct. The correct answer is {correct_answer}.')  
                elif answer:  
                    incorrect_count += 1  
                    stats_dict["ex"] = 0
                    print(f'The answer for path "{path}" is incorrect. The correct answer is {correct_answer}, but the provided answer is {answer}.')  
            else:  
                incorrect_count += 1
                stats_dict["ex"] = 0
                print(f'Could not find reference_answer.txt file at path "{path}".')  

        if save_flg:
            eval_jaon_list.append(stats_dict)
        else:
            eval_jaon_list[change_idx] = stats_dict

        with open(os.path.join(path,"eval_stats.json"), "w") as f_eval:
            json.dump(eval_jaon_list, f_eval, indent=4)

    return data, total_questions, correct_count, incorrect_count


def complement_AIR(ref_json, llm_response, setting, model_use, total_questions, correct_count, incorrect_count):
    for root, _ in ref_json.items(): 
        if root not in llm_response:
            try:
                with open(os.path.join(root,"eval_stats.json"), "r") as f_eval:
                    eval_jaon_list = json.load(f_eval)
                    eval_jaon_list = [json.loads(i) for i in set(json.dumps(d, sort_keys=True) for d in eval_jaon_list)]
            except FileNotFoundError:
                continue
            
            save_flg = True
            if eval_jaon_list:
                for i in range(len(eval_jaon_list)):
                    dict_i = eval_jaon_list[i]
                    if dict_i["setting"] == setting and dict_i["model"] == model_use:
                        save_flg = False
                        change_idx = i
                        break

            dict_miss = {}
            for i in range(len(eval_jaon_list)):
                dict_i = eval_jaon_list[i]
                if dict_i["setting"] == "REACT" and dict_i["model"] == model_use:
                    dict_agent = dict_i
                
            print(root)
            dict_miss = dict_agent.copy()
            dict_miss["setting"] = "AIR"
            
            if save_flg:
                eval_jaon_list.append(dict_miss)
            else:
                eval_jaon_list[change_idx] = dict_miss
            with open(os.path.join(root,"eval_stats.json"), "w") as f_eval:
                json.dump(eval_jaon_list, f_eval, indent=4)
            
            processed_output = dict_miss["ex"]
            if processed_output == 1:
                correct_count += 1
            else:
                incorrect_count += 1

            total_questions += 1

    return total_questions, correct_count, incorrect_count


def print_stats(total_questions, correct_count, incorrect_count):
    # Calculate and print statistics  
    if total_questions != correct_count + incorrect_count:
        print(total_questions)
        print("Wrong case counting!!!")
        # raise AssertionError
    accuracy_rate = (correct_count / total_questions) * 100 if total_questions else 0.0  
    
    print("\n======================== Statistics: ========================")  
    print(f"Total questions: {total_questions}")  
    print(f"Correct answers: {correct_count}")  
    print(f"Incorrect answers: {incorrect_count}")  
    print(f"Accuracy rate: {accuracy_rate:.2f}%")  
    print("================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--llm_response_path", type=str, required=True, help="Path to the llm response directory.")
    parser.add_argument("--ref_response_path", type=str, required=True, help="Path to the reference llm response directory for AIR (i.e., agent setting).")
    parser.add_argument("--action_name", type=str, default="None", help="The action name in action setting.")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4-32k", required=True, help="Model name of chosen LLM.")
    parser.add_argument("--model_version", type=str, default="base", choices=['base', 'agent', 'inter_agent'], required=True, help="Model version of chosen LLM.")
    args = parser.parse_args()

    if args.model_version == "inter_agent":
        setting = "AIR"
    elif args.model_version == "agent":
        if "plotqa" in args.action_name:
            setting = "Agent"
        else:
            setting = "REACT"
    else:
        setting = "base"

    with open(args.ref_response_path, "r") as f_json:
        ref_json = json.load(f_json)

    llm_response, total_questions, correct_count, incorrect_count = eval_main(args.llm_response_path, setting, args.llm_model_name, args.action_name)
    
    if args.model_version == 'inter_agent':
        total_questions, correct_count, incorrect_count = complement_AIR(ref_json, llm_response, setting, args.llm_model_name, total_questions, correct_count, incorrect_count)
    
    print_stats(total_questions, correct_count, incorrect_count)
