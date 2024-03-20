import sys
from io import StringIO
import contextlib
import json
import os
import contextlib
import ast
from collections import Counter
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

@contextlib.contextmanager  
def capture_output():  
    new_out, new_err = StringIO(), StringIO()  
    old_out, old_err = sys.stdout, sys.stderr  
    try:  
        sys.stdout, sys.stderr = new_out, new_err  
        yield sys.stdout, sys.stderr  
    finally:  
        sys.stdout, sys.stderr = old_out, old_err 


def is_float(s):
    if '.' not in s:
        return s.isdecimal()
    else:
        try:
            before_decimal, after_decimal = s.split('.')
        except ValueError:
            print(s)
            print(root)
            raise AssertionError

        return before_decimal.isdecimal() and (after_decimal.isdecimal() or after_decimal == '')


def parse_chain(node, prefix=[]):
    if isinstance(node, ast.Name):
        return [node.id] + prefix
    elif isinstance(node, ast.Call):
        return parse_chain(node.func, prefix)
    elif isinstance(node, ast.Attribute):
        return parse_chain(node.value, [node.attr] + prefix)
    else:
        return []

def parse(node):
    results = []
    if isinstance(node, (ast.Call, ast.Attribute)):
        chain = parse_chain(node)
        if chain:
            results.append('.'.join(chain))
    for field in getattr(node, '_fields', []):
        value = getattr(node, field)
        if isinstance(value, list):
            for item in value:
                results.extend(parse(item))
        elif isinstance(value, ast.AST):
            results.extend(parse(value))
    return results

def calc_private_func_recall(code_gen,gt_list):
    code_gen = code_gen.replace('"""', "'''")
    results = parse(ast.parse(code_gen))
    results = set(results)
    score = 0
    for func in results:
        if func in gt_list:
            score += 1

    return score / len(gt_list)

def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0


def eval_dict_initialize(root, setting, model_use, action_name):
    eval_jaon_list = []
    try:
        with open(os.path.join(root,"eval_stats.json"), "r") as f_eval:
            eval_jaon_list = json.load(f_eval)
            eval_jaon_list = [json.loads(i) for i in set(json.dumps(d, sort_keys=True) for d in eval_jaon_list)]
    except:
        pass
    
    # check:
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

    if is_folder_empty(os.path.join(root,"pred_result")):
        stats_dict["valid"] = False
    if "analysis" in root or "conclude" in root or "una_" in root or "bg_" in root:
        stats_dict["is_code"] = False

    if "short" in root:
        stats_dict["class"].append("short")
    else:
        stats_dict["class"].append("long")

    if "private" in root:
        stats_dict["class"].append("private")
    else:
        stats_dict["class"].append("normal")

    if action_name in root:
        stats_dict["class"].append(action_name)

    return eval_jaon_list, stats_dict, save_flg, change_idx


def exec_code(root):
    os.chdir(root)
    with capture_output() as (out, err):
        try:
            with open('eval.py') as f:
                content = f.read()
                exec(content)
        except Exception as e:
            print(f"Error occurred while running eval.py in {root}: {e}")

    stdout, _ = out.getvalue(), err.getvalue()
    processed_output = stdout.strip()
    return processed_output


def format_exec_out(processed_output, root):
    if "\n" in processed_output:
        processed_output = processed_output.split("\n")
        for i in range(len(processed_output)):
            line = processed_output[i]
            if line != "True" and line != "False":
                if is_float(line):
                    if float(line) > 0.6:
                        processed_output[i] = "True"
                    else:
                        processed_output[i] = "False"
                else:
                    print("Wrong Output EVAL!!!   ", line)
                    print(root)
                    processed_output[i] = "False"
    else:
        if processed_output != "True" and processed_output != "False":
            if is_float(processed_output):
                if float(processed_output) > 0.6:
                    processed_output = "True"
                else:
                    processed_output = "False"
            else:
                print("Wrong Eval Output: ", processed_output)
                print(root)
                processed_output = "False"

    return processed_output


def calc_private_recall_main(root, code_seg_file):
    with open(os.path.join(root, code_seg_file), "r") as f_c:
        code_gen = f_c.read()

    with open(os.path.join(root, 'ref_code.py'), "r") as f_p:
        ref_code = f_p.read()

    func_gt = ""
    for line in ref_code.split("\n"):
        if "from decision_company import" in line:
            func_gt = line
            break
    func_gt = func_gt.replace("from decision_company import ", "")
    private_func_gt = []
    for func in func_gt.split(","):
        private_func_gt.append(func.strip())

    try:
        privat_func_score = calc_private_func_recall(code_gen, private_func_gt)
    except:
        privat_func_score = 0
    if privat_func_score > 1:
        print(root)
        print(code_gen)
        print(privat_func_score)
        raise AssertionError
    
    return privat_func_score


def calc_numerical_score(processed_output, intent_num, results, privat_func_score, score_all):
    if isinstance(processed_output, list):
        if len(processed_output) != intent_num:
            processed_output = ["False"] * intent_num
        real_score = []
        results.extend(processed_output)
        for i in processed_output:
            if i == "True":
                print(privat_func_score)
                score_all.append(privat_func_score)
                real_score.append(privat_func_score)
            else:
                real_score.append(0)

    else:
        if intent_num > 1:
            processed_output = ["False"] * intent_num
            results.extend(processed_output)
        else:
            results.append(processed_output)
        if processed_output == "True":
            print(privat_func_score)
            score_all.append(privat_func_score)
            real_score = [privat_func_score]
        else:
            real_score = [0]

    return results, real_score


def eval_main(llm_response, code_seg_file, setting, model_use, action_name):
    results = []
    score_all = []
    total_questions = 0
    for root, _ in llm_response.items(): 
        eval_jaon_list, stats_dict, save_flg, change_idx = eval_dict_initialize(root, setting, model_use, action_name)
        total_questions += 1
        with open(os.path.join(root, 'meta_data.json'), "r") as f_c:
            meta_data = json.load(f_c)

        if isinstance(meta_data["result_type"], list):
            intent_num = len(meta_data["result_type"])
            total_questions = total_questions + intent_num - 1
        else:
            intent_num = 1

        processed_output = exec_code(root)
        processed_output = format_exec_out(processed_output, root)

        if "private" in root and "True" in processed_output:
            privat_func_score = calc_private_recall_main(root, code_seg_file)
        else:
            privat_func_score = 1

        results, real_score = calc_numerical_score(processed_output, intent_num, results, privat_func_score, score_all)

        stats_dict["ex"] = processed_output
        stats_dict["exr"] = real_score

        if "True" in processed_output:
            print(root)

        if save_flg:
            eval_jaon_list.append(stats_dict)
        else:
            eval_jaon_list[change_idx] = stats_dict

        with open(os.path.join(root,"eval_stats.json"), "w") as f_eval:
            json.dump(eval_jaon_list, f_eval, indent=4)

    return results, score_all, total_questions


def load_format_eval_dicts(root):
    with open(os.path.join(root,"eval_stats.json"), "r") as f_eval:
        eval_jaon_list = json.load(f_eval)
        eval_jaon_list = [json.loads(i) for i in set(json.dumps(d, sort_keys=True) for d in eval_jaon_list)]
        for dic in eval_jaon_list:
            if isinstance(dic["ex"], list):
                for i in range(len(dic["ex"])):
                    item = dic["ex"][i]
                    if item != "True":
                        dic["exr"][i] = 0
            else:
                if dic["ex"] != "True":
                    dic["exr"][0] = 0

    return eval_jaon_list


def complement_AIR(ref_json, llm_response, setting, model_use, results, score_all, total_questions):
    for root, _ in ref_json.items(): 
        if root not in llm_response:
            eval_jaon_list = load_format_eval_dicts(root)

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
                if dict_i["setting"] == "COT" and dict_i["model"] == model_use:
                    dict_cot = dict_i

            dict_miss = dict_cot.copy()
            dict_miss["setting"] = "AIR"
            
            if save_flg:
                eval_jaon_list.append(dict_miss)
            else:
                eval_jaon_list[change_idx] = dict_miss
            with open(os.path.join(root,"eval_stats.json"), "w") as f_eval:
                json.dump(eval_jaon_list, f_eval, indent=4)
            
            processed_output = dict_miss["ex"]
            privat_func_score = dict_miss["exr"]

            total_questions += 1
            with open(os.path.join(root, 'meta_data.json'), "r") as f_c:
                meta_data = json.load(f_c)

            if isinstance(meta_data["result_type"], list):
                intent_num = len(meta_data["result_type"])
                total_questions = total_questions + intent_num - 1
            else:
                intent_num = 1
                
            if isinstance(processed_output, list):
                if len(processed_output) != intent_num:
                    processed_output = ["False"] * intent_num
                results.extend(processed_output)
                for i in range(len(processed_output)):
                    if processed_output[i] == "True":
                        print(root)
                        score_all.append(privat_func_score[i])
            else:
                if intent_num > 1:
                    processed_output = ["False"] * intent_num
                    results.extend(processed_output)
                else:
                    results.append(processed_output)
                if processed_output == "True":
                    score_all.extend(privat_func_score)

    return results, score_all, total_questions

def print_stats(results, total_questions, score_all):
    stats = Counter(results)
    true_cnt = stats["True"]
    accuracy_rate = true_cnt / total_questions
    total_score = sum(score_all)
    score_rate = total_score / total_questions

    print("\n================= Statistics: =================")
    print(f"Total questions: {total_questions}")
    print(f"Stats: {stats}")
    print(f"Accuracy rate: {accuracy_rate:.4f}")
    print("---------------------------------------------------")
    print((f"Score: {total_score:.4f}"))
    print((f"Score Rate: {score_rate:.4f}"))
    print("===================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--llm_response_path", type=str, required=True, help="Path to the llm response directory.")
    parser.add_argument("--ref_response_path", type=str, required=True, help="Path to the reference llm response directory for AIR (i.e., agent setting).")
    parser.add_argument("--code_seg_fn", type=str, required=True, help="The filename of the prediction code segment.")
    parser.add_argument("--action_name", type=str, default="None", help="The action name in action setting.")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4-32k", required=True, help="Model name of chosen LLM.")
    parser.add_argument("--model_version", type=str, default="base", choices=['base', 'agent', 'inter_agent'], required=True, help="Model version of chosen LLM.")
    args = parser.parse_args()

    if args.model_version == "inter_agent":
        setting = "AIR"
    elif args.model_version == "agent":
        setting = "COT"
    else:
        setting = "base"

    with open(args.ref_response_path, "r") as f_json:
        ref_json = json.load(f_json)

    with open(args.llm_response_path, "r") as f_json:
        llm_response = json.load(f_json)

    results, score_all, total_questions = eval_main(llm_response, args.code_seg_fn, setting, args.llm_model_name, args.action_name)

    if args.model_version == 'inter_agent':
        results, score_all, total_questions = complement_AIR(ref_json, llm_response, setting, args.llm_model_name, results, score_all, total_questions)
    
    print_stats(results, total_questions, score_all)

