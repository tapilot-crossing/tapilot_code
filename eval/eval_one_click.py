import json
import os
import numpy as np
import argparse

def is_penultimate_directory(path):  
    """  
    This function checks if the given path is a penultimate directory (i.e., its subdirectories do not contain any other directories).  
  
    :param path: Path to the directory to check  
    :return: True if the directory is penultimate, False otherwise  
    """  
    # List all subdirectories in the given path  
    subdirectories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]  
  
    # If there are no subdirectories, the given path is not a penultimate directory  
    if not subdirectories:  
        return False  
  
    # Check each subdirectory to see if it contains any other directories  
    for subdir in subdirectories:  
        subdir_path = os.path.join(path, subdir)  
        subdir_subdirectories = [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))]  
  
        # If a subdirectory contains other directories, the given path is not a penultimate directory  
        if subdir_subdirectories:  
            return False  
  
    # If none of the subdirectories contain other directories, the given path is a penultimate directory  
    return True 

def eval_main(model_use, target_path, setting):
    score_all = []
    score_normal = []
    score_action = []
    score_private = []
    score_pri_act = []
    score_code = []
    score_mc = []

    for root, _, _ in os.walk(target_path): 
        if is_penultimate_directory(root): 
            if "__pycache__" in root or "description_table" in root or "mid_results" in root or "src" in root:
                continue

            try:
                with open(os.path.join(root,"eval_stats.json"), "r") as f_json:
                    eval_stats = json.load(f_json)
            except FileNotFoundError:
                print("!!! Not yet evaluate: ", root)
                continue


            if "analysis" in root or "conclude" in root or "una" in root or "bg" in root or "plotqa" in root:
                for dic in eval_stats:
                    if dic["model"] != model_use:
                        continue

                    if dic["setting"] == setting:
                        score_mc.append(dic["ex"])
                        score_all.append(dic["ex"])

            else:
                for dic in eval_stats:
                    if dic["model"] != model_use:
                        continue
                    if isinstance(dic["ex"], list):
                        for i in range(len(dic["ex"])):
                            if dic["ex"][i] == "False":
                                dic["exr"][i] = 0
                    else:
                        if dic["ex"] == "False":
                            dic["exr"][0] = 0

                    if dic["setting"] == setting:
                        score_code.extend(dic["exr"])
                        score_all.extend(dic["exr"])


            if "private" not in root and "action" not in root:
                for dic in eval_stats:
                    if dic["model"] != model_use:
                        continue

                    if (dic["setting"] == setting) and dic["model"] == model_use:
                        score_normal.extend(dic["exr"])


            if "private" in root and "action" not in root:
                for dic in eval_stats:
                    if dic["model"] != model_use:
                        continue

                    if (dic["setting"] == setting) and dic["model"] == model_use:
                        score_private.extend(dic["exr"])


            if "private" in root and "action" in root:
                for dic in eval_stats:
                    if dic["model"] != model_use:
                        continue
                    if "analysis" in root or "conclude" in root or "una" in root or "bg" in root or "plotqa" in root:
                        if dic["setting"] == setting:
                            score_pri_act.append(dic["ex"])

                    else:
                        if dic["model"] != model_use:
                            continue

                        if dic["setting"] == setting:
                            score_pri_act.extend(dic["exr"])

            if "private" not in root and "action" in root:
                for dic in eval_stats:
                    if dic["model"] != model_use:
                        continue
                    if "analysis" in root or "conclude" in root or "una" in root or "bg" in root or "plotqa" in root:
                        if dic["setting"] == setting:
                            score_action.append(dic["ex"])

                    else:
                        if dic["model"] != model_use:
                            continue

                        if dic["setting"] == setting:
                            score_action.extend(dic["exr"])

    value_score_all = np.sum(score_all) / len(score_all)
    value_score_mc = np.sum(score_mc) / len(score_mc)
    value_score_code = np.sum(score_code) / len(score_code)
    value_score_pri_act = np.sum(score_pri_act) / len(score_pri_act)
    value_score_private = np.sum(score_private) / len(score_private)
    value_score_action = np.sum(score_action) / len(score_action)
    value_score_normal = np.sum(score_normal) / len(score_normal)

    print("================================================== Result ==================================================")
    # Define categories and scores, multiplying each by 100 and formatting to one decimal place.
    categories = ["Normal", "Action", "Private", "Pri-Act", "Code", "Multi-choice", "Overall"]
    scores = [
        value_score_normal * 100,
        value_score_action * 100,
        value_score_private * 100,
        value_score_pri_act * 100,
        value_score_code * 100,
        value_score_mc * 100,
        value_score_all * 100
    ]

    # Print the header row with categories. Ensure everything is centered and the space is reduced.
    # Additionally, insert separators after 'Pri-Act' and 'Multi-choice'.
    print("{:^13}".format(""), end="")
    for i, category in enumerate(categories):
        end_char = "| " if category in ["Pri-Act", "Multi-choice"] else ""
        print("{:^13}".format(category), end=end_char)
    print()  # Move to the next line after printing all categories.

    # Print the "Inter-Agent" row with scores. Ensure each score is centered and formatted to one decimal place.
    # Similarly, insert separators after the scores corresponding to 'Pri-Act' and 'Multi-choice'.
    print("{:^13}".format(setting), end="")
    for i, score in enumerate(scores):
        category = categories[i]
        end_char = "| " if category in ["Pri-Act", "Multi-choice"] else ""
        print("{:^13.1f}".format(score), end=end_char)
    print()  # Ensure there's a newline at the end of the row.
    print("============================================================================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to tapilot data.")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4-32k", required=True, help="Model name of chosen LLM.")
    parser.add_argument("--model_version", type=str, default="base", choices=['base', 'agent', 'inter_agent'], required=True, help="Model version of chosen LLM.")
    args = parser.parse_args()

    eval_main(args.llm_model_name, args.data_path, args.model_version)



