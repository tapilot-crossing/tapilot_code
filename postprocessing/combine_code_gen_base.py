import json
import os
import argparse
from utils import format_code_path

def prepare_private_funcs(root, llm_response_dict_func_02):
    private_func = llm_response_dict_func_02[root]
    cut_idx_1 = private_func.rfind("<FUNCTIONS>")
    cut_idx_2 = private_func.rfind("</FUNCTIONS>")

    if cut_idx_1 == -1:
        private_func = private_func[:cut_idx_2]
    else:
        private_func = private_func[cut_idx_1:cut_idx_2].replace("<FUNCTIONS>", "")

    return private_func

def prepare_code_head(root, llm_response_dict_func_02):
    with open(os.path.join(root, 'prompt_curr.txt'), "r") as f_p:
        prompt_curr = f_p.read()

    private_func = ""
    if "private" in root:
        private_func = prepare_private_funcs(root, llm_response_dict_func_02)

    cut_idx = prompt_curr.rfind("[YOU (AI assistant)]")
    if cut_idx != -1:
        prompt_curr = prompt_curr[cut_idx:]

    cut_idx_2 = prompt_curr.rfind("'''")
    if cut_idx_2 != -1:
        code_head = prompt_curr[cut_idx_2:].replace("'''", "")
    else:
        raise AssertionError
    
    return private_func, code_head

def extract_code(llm_resp, code_head):
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

def main(llm_response_path, private_func_path, code_seg_fn, code_pred_fn):
    with open(llm_response_path, "r") as f_json:
        llm_response = json.load(f_json)

    with open(private_func_path, 'r') as f_r:  
        llm_response_dict_func_02 = json.load(f_r) 


    for root, llm_resp in llm_response.items(): 
        code_base_path = os.path.join(root, 'src/ref_code_hist.py')
        with open(code_base_path, "r") as f_code:
            code_base = f_code.read()

        private_func, code_head = prepare_code_head(root, llm_response_dict_func_02)
        code_add = extract_code(llm_resp, code_head)
        
        if "short_" not in root:
            if "turn_1" not in root:
                for line in code_add.split("\n"):
                    if "def read_csv_file(" in line:
                        continue
                    if "read_csv_file(" in line or "pd.read_csv(" in line:
                        code_add = code_add.replace(line, "")
                        break
            else:
                code_add = format_code_path(code_add)
        else:
            if "turn_1_short_1" not in root:
                for line in code_add.split("\n"):
                    if "def read_csv_file(" in line:
                        continue
                    if "read_csv_file(" in line or "pd.read_csv(" in line:
                        code_add = code_add.replace(line, "")
                        break
            else:
                code_add = format_code_path(code_add)

        code_add = code_add.replace("</code>", "").replace("<code>", "")
        if "private" in root:
            for line in code_add.split("\n"):
                if "from decision_company import read_csv_file," in line:
                    code_add = code_add.replace(line, "from decision_company import " + private_func).replace("# please import the necessary private functions from decision_company first", "")

        with open(os.path.join(root, code_seg_fn), "w") as f_out:
            f_out.write(code_add)

        code_all = code_base + "\n\n" + "import os\n" + code_add
        for line in code_all.split("\n"):
            if "plt.show(" in line or "show_plots(" in line:
                code_all = code_all.replace(line, "")

        with open(os.path.join(root, code_pred_fn), "w") as f_out:
            f_out.write(code_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories and update databases.")
    parser.add_argument("--llm_response_path", type=str, required=True, help="Path to the llm response directory.")
    parser.add_argument("--private_func_path", type=str, required=True, help="Path to the llm response directory of private function extraction.")
    parser.add_argument("--code_seg_fn", type=str, required=True, help="The filename of the prediction code segment.")
    parser.add_argument("--code_pred_fn", type=str,required=True, help="The filename of the prediction code.")
    args = parser.parse_args()

    main(args.llm_response_path, args.private_func_path, args.code_seg_fn, args.code_pred_fn)