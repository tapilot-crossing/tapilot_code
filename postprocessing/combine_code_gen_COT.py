import argparse
from utils import format_code_path
import json
import os
import numpy as np

def prepare_code_head(root):
    with open(os.path.join(root, 'prompt_curr.txt'), "r") as f_p:
        prompt_curr = f_p.read()

    cut_idx = prompt_curr.rfind("[YOU (AI assistant)]")
    if cut_idx != -1:
        prompt_curr = prompt_curr[cut_idx:]

    cut_idx_2 = prompt_curr.rfind("'''")
    if cut_idx_2 != -1:
        code_head = prompt_curr[cut_idx_2+3:]
    else:
        code_head = ""
    
    return code_head

def remove_read_csv(code_add):
    for line in code_add.split("\n"):
        if "def read_csv_file(" in line:
            continue
        if "read_csv_file(" in line or "pd.read_csv(" in line:
            code_add = code_add.replace(line, "")

    return code_add

def extract_python_code(llm_resp, code_head, root):
    if "---BEGIN " in llm_resp:
        cut_idx = llm_resp.find("---BEGIN ")
        code_gen = llm_resp[cut_idx:]
        
        cut_idx_2 = code_gen.find("import ")
        if "from " in code_gen[:cut_idx_2]:
            print(root)
            raise AssertionError
        code_add = code_gen[cut_idx_2:]

        cut_idx_3 = code_add.rfind("```")
        if cut_idx_3 == -1:
            cut_idx_3 = code_add.rfind("---END ")
            if cut_idx_3 == -1:
                cut_idx_3 = code_add.rfind("'''")
        code_add = code_add[:cut_idx_3]

    elif "```python" in llm_resp:
        cut_idx = llm_resp.find("```python")
        code_gen = llm_resp[cut_idx:].replace("```python", "")
        
        cut_idx_2 = code_gen.find("import ")
        if "from " in code_gen[:cut_idx_2]:
            print(root)
            raise AssertionError
        code_add = code_gen[cut_idx_2:]
        cut_idx_3 = code_add.rfind("```")
        if cut_idx_3 == -1:
            cut_idx_3 = code_add.rfind("---END CODE TEMPLATE---")
            if cut_idx_3 == -1:
                cut_idx_3 = code_add.rfind("'''")
        code_add = code_add[:cut_idx_3]

    else:
        if "import " in llm_resp:
            code_gen = llm_resp
            cut_idx_2 = code_gen.find("import ")

            code_add = code_gen[cut_idx_2:]
            cut_idx = code_add.rfind("```")
            if cut_idx != -1:
                code_add = code_add[:cut_idx]
            
            code_add = code_add.replace("\n'''", "").replace("\n---END CODE TEMPLATE---", "").replace("\n```", "")
        else:
            cut_idx = llm_resp.rfind("'''")
            code_add = code_head + llm_resp[:cut_idx]
            code_add = code_add.replace("'''", "").replace("---END CODE TEMPLATE---", "").replace("```", "")

    return code_add

def main(llm_response_path, code_seg_fn, code_pred_fn):
    with open(llm_response_path, "r") as f_json:
        llm_response = json.load(f_json)

    for root, llm_resp in llm_response.items(): 
        code_base_path = os.path.join(root, 'reference/ref_code_hist.py')
        with open(code_base_path, "r") as f_code:
            code_base = f_code.read()

        code_head = prepare_code_head(root)
        code_add = extract_python_code(llm_resp, code_head, root)
        
        code_add = code_add.replace("---END CODE TEMPLATE---", "").replace("```", "")
        cut_idx = code_add.rfind("pickle.dump(")
        if cut_idx == -1:
            cut_idx = code_add.rfind("plt.")
            if cut_idx == -1:
                cut_idx = code_add.rfind("save_plot(")
                cut_idx_tmp = code_add.rfind("save_plot(filename, dpi=100")
                if cut_idx == cut_idx_tmp:
                    cut_idx = len(code_add) - int(np.ceil(len(code_add)/50))

        code_end = code_add[cut_idx:]
        code_end_clr = code_end.replace("'''", "")
        code_add = code_add.replace(code_end, code_end_clr)
        change_flg = False
        for line in code_add.split("\n"):
            if "pickle.load(" in line:
                code_add = code_add.replace(line, "")
                
        if "short_" not in root:
            if "turn_1" not in root:
                code_add = remove_read_csv(code_add)
            else:
                code_add = format_code_path(code_add)
        else:
            if "turn_1_short_1" not in root:
                code_add = remove_read_csv(code_add)
            else:
                code_add = format_code_path(code_add)

        code_add = code_add.replace("</code>", "").replace("<code>", "")

        with open(os.path.join(root, code_seg_fn), "w") as f_out:
            f_out.write(code_add)

        code_all = code_base + "\n\n" + code_add
        for line in code_all.split("\n"):
            if "plt.show(" in line or "show_plots(" in line:
                code_all = code_all.replace(line, "")

        with open(os.path.join(root, code_pred_fn), "w") as f_out:
            f_out.write(code_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--llm_response_path", type=str, required=True, help="Path to the llm response directory.")
    parser.add_argument("--code_seg_fn", type=str, required=True, help="The filename of the prediction code segment.")
    parser.add_argument("--code_pred_fn", type=str,required=True, help="The filename of the prediction code.")
    args = parser.parse_args()

    main(args.llm_response_path, args.code_seg_fn, args.code_pred_fn)