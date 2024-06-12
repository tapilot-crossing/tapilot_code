import json
import os
import re
from tqdm import tqdm 
from prompts import pre_prompt_func_names_user, pre_prompt_func_names_AI, prompt_AIR_02_curr, prompt_AIR_teacher_user, prompt_AIR_teacher_AI, private_func_names_AIR_01, private_func_names_AIR_02, prompt_react_code_hist_analysis_credit, prompt_react_code_hist_analysis_credit_AIR, prompt_react_code_hist_analysis_ATP, prompt_react_code_hist_analysis_ATP_AIR, prompt_react_code_hist_analysis_melb_AIR, prompt_react_code_hist_analysis_melb, prompt_react_code_hist_analysis_laptop, prompt_react_code_hist_analysis_laptop_AIR, prompt_react_code_hist_analysis_fastfood_AIR, prompt_react_code_hist_analysis_fastfood, laptop_df, fastfood_df, melb_df, atp_df, credit_df, prompt_AIR_student_AI, prompt_AIR_student_user
from utils import get_llm_response, process_python_output, capture_output, format_code, remove_pickle_code, find_last_template

class tapilot_agent:
    def __init__(self, data_dirs, output_path, llm_engine, model_version, decision_company, data_path):
        """
        Initializes the parameters of tapilot agent

        Paras:

        """
        self.data_dirs = data_dirs
        self.output_path = output_path
        self.llm_engine = llm_engine
        self.model_version = model_version
        self.decision_company = decision_company
        self.engine_name = llm_engine.replace("-", "_")
        self.role = 'assistant'
        self.data_path = data_path.replace("interaction_data", "resource")


    def predict_private_functions_base_agent(self):
        """
        Predict functions used in private lib mode. Model version:  
        """
        pred_private_functions_dict = {}
        count_save = 0
        save_dir = os.path.join(self.output_path, 'private/private_funcs_' + self.model_version +  '_' + self.engine_name + '.json')
        try:
            with open(save_dir, 'r') as f:
                pred_private_functions_dict = json.load(f)
        except:
            pass

        for root in tqdm(self.data_dirs, desc="Processing baseline private step 01"):
            if root in pred_private_functions_dict and pred_private_functions_dict[root] != "Failed!":
                continue
            if "private" not in root:
                continue

            count_save += 1
            with open(os.path.join(root, 'reference/prompt_code_hist.json'), 'r') as f_w:  
                prompt_code_hist_list = json.load(f_w)

            # The standard input format of GPTs: list of dicts which stores dialouge history
            prompt_code_hist_list[-2]["content"] = pre_prompt_func_names_user.format(dialogue_hist = prompt_code_hist_list[-2]["content"])
            prompt_code_hist_list[-1]["content"] = pre_prompt_func_names_AI.replace("[YOU (AI assistant)]:", "")

            message_obj = prompt_code_hist_list

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            message_llm = llm_response["choices"][0]["message"]["content"]

            pred_private_functions_dict[root] = message_llm

            if count_save % 10 == 0:
                with open(save_dir, "w") as f_out:
                    json.dump(pred_private_functions_dict, f_out, indent = 4)

        with open(save_dir, 'w') as f:
            json.dump(pred_private_functions_dict, f, indent=4)

        return save_dir


    def code_gen_base_agent(self):
        func_dir = self.predict_private_functions_base_agent()
        with open(func_dir, 'r') as f:
            pred_private_functions_dict = json.load(f)

        llm_response_dict = {}
        count_save = 0
        count_all = 0
        save_dir = os.path.join(self.output_path, 'normal/code_gen_' + self.model_version +  '_' + self.engine_name + '.json')
        try:
            with open(save_dir, 'r') as f:
                llm_response_dict = json.load(f)
        except:
            pass

        for root in tqdm(self.data_dirs, desc="Processing baseline all step 02"):
            if root in llm_response_dict and llm_response_dict[root] != "Failed!":
                continue

            count_all += 1
            count_save += 1
            with open(os.path.join(root, 'reference/prompt_code_hist.json'), 'r') as f_w:  
                prompt_code_hist_list = json.load(f_w)
                if self.model_version == "base":
                    prompt_code_hist_list[-1]["content"] = ""
                elif self.model_version == "agent":
                    prompt_code_hist_list[-1]["content"] = "I need first to write a step-by-step outline and then write the code:"
                
            # Extract predicted private function
            if "private" in root:
                private_func = pred_private_functions_dict[root]
                cut_idx_1 = private_func.rfind("<FUNCTIONS>")
                cut_idx_2 = private_func.rfind("</FUNCTIONS>")

                if cut_idx_1 == -1 and cut_idx_2 == -1 and "import " in private_func:
                    private_func = "read_csv_file"
                elif cut_idx_1 == -1 or cut_idx_2 == -1:
                    private_func = private_func.replace("<FUNCTIONS>", "").replace("</FUNCTIONS>", "")
                else:
                    private_func = private_func[cut_idx_1:cut_idx_2].replace("<FUNCTIONS>", "")
                private_func_all = []
                for func in private_func.split(","):
                    private_func_all.append(func.strip())

                function_add = ""
                for func in private_func_all:
                    if func in self.decision_company:
                        function_add = function_add + self.decision_company[func] + "\n\n"

                # The standard input format of GPTs: list of dicts which stores dialouge history
                prompt_curr = prompt_code_hist_list[-2]["content"]
                prompt_curr_new = prompt_curr.replace("from decision_company import read_csv_file,", "from decision_company import " + private_func + "\n\n" + function_add).replace("# please import the necessary private functions from decision_company first", "")
                prompt_curr_new = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', prompt_curr_new)
                prompt_code_hist_list[-2]["content"] = prompt_curr_new
                
            message_obj = prompt_code_hist_list

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            message_llm = llm_response["choices"][0]["message"]["content"]

            llm_response_dict[root] = message_llm

            if count_save % 10 == 0:
                with open(save_dir, "w") as f_out:
                    json.dump(llm_response_dict, f_out, indent = 4)

        with open(save_dir, 'w') as f_out:
            json.dump(llm_response_dict, f_out, indent=4)


    def private_func_AIR(self):
        llm_response_dict_func_01 = {}
        prompt_dict_func_01 = {}
        count_save = 0

        private_postfix = "_AIR"
        private_prefix = "01_private_func_"
        save_dir = os.path.join(self.output_path, 'private/' + private_prefix + self.engine_name + private_postfix + '.json')
        save_prompt = os.path.join(self.output_path, 'private_prompt/' + private_prefix + self.engine_name + private_postfix + '.json')
        try:
            with open(save_dir, "r") as f_in:
                llm_response_dict_func_01 = json.load(f_in)
        except:
            pass

        try:
            with open(save_prompt, "r") as f_in:
                prompt_dict_func_01 = json.load(f_in)
        except:
            pass

        for root in tqdm(self.data_dirs, desc="Processing AIR Func Name 01"):
            if root in llm_response_dict_func_01 and llm_response_dict_func_01[root] != "Failed!":
                continue
            if "private" not in root:
                continue

            count_save += 1
            with open(os.path.join(root, "reference/prompt_code_hist.txt"), 'r') as f_j: 
                prompt_AIR = f_j.read()
                
            cut_idx_1 = prompt_AIR.find("[USER (data scientist)]")
            head_info = prompt_AIR[:cut_idx_1]
            cut_idx_2 = prompt_AIR.rfind("[USER (data scientist)]")

            # check if having dialouge history
            if cut_idx_1 == cut_idx_2:
                print("No Dialogue History: ", root)
                continue

            dialog_hist = prompt_AIR[cut_idx_1:cut_idx_2]

            private_func_names_AIR_01_now = private_func_names_AIR_01.format(head=head_info, dialogue_hist=dialog_hist)
            private_func_names_AIR_01_now = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', private_func_names_AIR_01_now)

            message_obj = [{"role": self.role, "content": private_func_names_AIR_01_now}]

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            llm_response = llm_response["choices"][0]["message"]["content"]

            llm_response_dict_func_01[root] = llm_response
            prompt_dict_func_01[root] = private_func_names_AIR_01_now

            if count_save % 10 == 0:
                with open(save_dir, "w") as f_out:
                    json.dump(llm_response_dict_func_01, f_out, indent = 4)
                
                with open(save_prompt, "w") as f_out:
                    json.dump(prompt_dict_func_01, f_out, indent = 4)

        with open(save_dir, "w") as f_out:
            json.dump(llm_response_dict_func_01, f_out, indent = 4)

        with open(save_prompt, "w") as f_out:
            json.dump(prompt_dict_func_01, f_out, indent = 4)

        return save_dir


    def private_func_inter_agent(self):

        private_prefix = "02_private_func_"
        private_postfix = "_AIR"

        save_dir = self.private_func_AIR()

        with open(save_dir, 'r') as f_r:  
            llm_response_dict_func_01 = json.load(f_r) 

        llm_response_dict_func_02 = {}
        prompt_dict_func_02 = {}
        count_save = 0

        save_dir_func = os.path.join(self.output_path, 'private/' + private_prefix + self.engine_name + private_postfix + '.json')
        save_prompt = os.path.join(self.output_path, 'private_prompt/' + private_prefix + self.engine_name + private_postfix + '.json')
        try:
            with open(save_dir_func, "r") as f_in:
                llm_response_dict_func_02 = json.load(f_in)
        except:
            pass

        try:
            with open(save_prompt, "r") as f_in:
                prompt_dict_func_02 = json.load(f_in)
        except:
            pass

        for root in tqdm(self.data_dirs, desc="Processing AIR Func Name 02"):
            if root in llm_response_dict_func_02 and llm_response_dict_func_02[root] != "Failed!":
                continue

            count_save += 1
            with open(os.path.join(root, "reference/prompt_code_hist.txt"), 'r') as f_j: 
                prompt_AIR = f_j.read()
                
            cut_idx_1 = prompt_AIR.find("[USER (data scientist)]")
            head_info = prompt_AIR[:cut_idx_1]
            cut_idx_2 = prompt_AIR.rfind("[USER (data scientist)]")
            dialog_hist = prompt_AIR[cut_idx_1:cut_idx_2]

            # check if having dialouge history
            if cut_idx_1 == cut_idx_2:
                print("No Dialogue History: ", root)
                continue
            cut_idx_3 = prompt_AIR.rfind("[YOU (AI assistant)]")
            new_query = prompt_AIR[cut_idx_2:cut_idx_3]

            cut_idx_3 = dialog_hist.rfind("[USER (data scientist)]")
            last_query = dialog_hist[cut_idx_3:]
            cut_idx_4 = last_query.find("'''")
            last_query = last_query[:cut_idx_4]

            try:
                example_learned = llm_response_dict_func_01[root]
                if "<pseudocode>" in example_learned:
                    cut_idx = example_learned.find("<pseudocode>")
                    example_learned = example_learned[cut_idx:].replace("<pseudocode>", "")
            except:
                continue
            
            private_func_names_AIR_02_now = private_func_names_AIR_02.format(head=head_info, dialogue_hist=dialog_hist, new_query=new_query, last_query=last_query, example=example_learned)
            private_func_names_AIR_02_now = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', private_func_names_AIR_02_now)

            message_obj = [{"role": self.role, "content": private_func_names_AIR_02_now}]

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            llm_response = llm_response["choices"][0]["message"]["content"]

            llm_response_dict_func_02[root] = llm_response
            prompt_dict_func_02[root] = private_func_names_AIR_02_now

            if count_save % 10 == 0:
                with open(save_dir_func, "w") as f_out:
                    json.dump(llm_response_dict_func_02, f_out, indent = 4)

                with open(save_prompt, "w") as f_out:
                    json.dump(prompt_dict_func_02, f_out, indent = 4)

        with open(save_dir_func, "w") as f_out:
            json.dump(llm_response_dict_func_02, f_out, indent = 4)

        with open(save_prompt, "w") as f_out:
            json.dump(prompt_dict_func_02, f_out, indent = 4)

        return save_dir_func


    def code_gen_AIR(self):
        llm_response_dict_AIR_01 = {}
        prompt_dict_AIR_01 = {}
        count_save = 0

        normal_prefix = "01_code_gen_AIR_"

        save_dir_pseudo = os.path.join(self.output_path, 'normal/' + normal_prefix + self.engine_name + '.json')
        save_prompt = os.path.join(self.output_path, 'normal_prompt/' + normal_prefix + self.engine_name+ '.json')

        try:
            with open(save_dir_pseudo, "r") as f_in:
                llm_response_dict_AIR_01 = json.load(f_in)
        except:
            pass

        try:
            with open(save_prompt, "r") as f_out:
                prompt_dict_AIR_01 = json.load(f_out)
        except:
            pass

        for root in tqdm(self.data_dirs, desc="Processing AIR normal 01"):
            if root in llm_response_dict_AIR_01 and llm_response_dict_AIR_01[root] != "Failed!":
                continue

            count_save += 1
            # The standard input format of GPTs: list of dicts which stores dialouge history
            with open(os.path.join(root, "reference/prompt_code_hist.json"), 'r') as f_j: 
                prompt_AIR = json.load(f_j)
                dial_hist_teacher = prompt_AIR[:-2]
                try:
                    if dial_hist_teacher[-2]["role"] != "user":
                        print("Wrong Interaction History!")
                        print(root)
                        raise AssertionError
                    
                    if dial_hist_teacher[-1]["role"] != "assistant":
                        print("Wrong Interaction History!")
                        print(root)
                        raise AssertionError

                except IndexError:
                    continue

            # remove all pickle dump in history
            for i in range(len(dial_hist_teacher)):
                for line in dial_hist_teacher[i]["content"].split("\n"):
                    if 'print(' in line or 'pickle.dump' in line:
                        dial_hist_teacher[i]["content"] = dial_hist_teacher[i]["content"].replace(line, "")

            teacher_user = {"role": "user"}
            teacher_user["content"] = prompt_AIR_teacher_user
            teacher_AI = {"role": "assistant"}
            teacher_AI["content"] = prompt_AIR_teacher_AI
            dial_hist_teacher.append(teacher_user)
            dial_hist_teacher.append(teacher_AI)

            message_obj = dial_hist_teacher

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            llm_response = llm_response["choices"][0]["message"]["content"]

            llm_response_dict_AIR_01[root] = llm_response
            prompt_dict_AIR_01[root] = dial_hist_teacher
            if count_save % 10 == 0:
                with open(save_dir_pseudo, "w") as f_out:
                    json.dump(llm_response_dict_AIR_01, f_out, indent = 4)

                with open(save_prompt, "w") as f_out:
                    json.dump(prompt_dict_AIR_01, f_out, indent = 4)

        with open(save_dir_pseudo, "w") as f_out:
            json.dump(llm_response_dict_AIR_01, f_out, indent = 4)

        with open(save_prompt, "w") as f_out:
            json.dump(prompt_dict_AIR_01, f_out, indent = 4)

        return save_dir_pseudo

    
    def code_gen_inter_agent(self):
        save_dir_func = self.private_func_inter_agent()
        with open(save_dir_func, 'r') as f_r:  
            llm_response_dict_func_02 = json.load(f_r) 

        save_dir_pseudo = self.code_gen_AIR()
        with open(save_dir_pseudo, 'r') as f_r:  
            llm_response_dict_AIR_pseudo_01 = json.load(f_r) 

        llm_response_dict_curr = {}
        prompt_dict_curr = {}
        count_save = 0

        normal_prefix = "02_code_gen_AIR_"
        save_dir = os.path.join(self.output_path, 'normal/' + normal_prefix + self.engine_name + '.json')
        save_prompt = os.path.join(self.output_path, 'normal_prompt/' + normal_prefix + self.engine_name + '.json')

        try:
            with open(save_dir, "r") as f_in:
                llm_response_dict_curr = json.load(f_in)
        except:
            pass

        try:
            with open(save_prompt, "r") as f_out:
                prompt_dict_curr = json.load(f_out)
        except:
            pass

        for root, llm_resp in tqdm(llm_response_dict_AIR_pseudo_01.items(), desc="Processing AIR normal 02"):
            if root in llm_response_dict_curr and llm_response_dict_curr[root] != "Failed!":
                continue

            count_save += 1
            with open(os.path.join(root, "reference/prompt_code_hist.txt"), 'r') as f_j: 
                prompt_hist = f_j.read()

            if "private" in root:
                private_func = llm_response_dict_func_02[root]
                cut_idx_1 = private_func.rfind("<FUNCTIONS>")
                cut_idx_2 = private_func.rfind("</FUNCTIONS>")

                if cut_idx_1 == -1 or cut_idx_2 == -1:
                    print("NO private Functions extracted!!!  ", root)
                    private_func_all = []
                    private_func_all.append("read_csv_file")
                else:
                    private_func_all = []
                    private_func = private_func[cut_idx_1:cut_idx_2].replace("<FUNCTIONS>", "")
                    for func in private_func.split(","):
                        private_func_all.append(func.strip())

                function_add = ""
                for func in private_func_all:
                    if func in self.decision_company:
                        function_add = function_add + self.decision_company[func] + "\n\n"
               
            cut_idx_1 = prompt_hist.rfind("[USER (data scientist)]")
            dial_hist = prompt_hist[:cut_idx_1]
            cut_info = dial_hist.find("[USER (data scientist)]")
            head_info = dial_hist[:cut_info]
            dial_hist = dial_hist[cut_info:]
            new_query = prompt_hist[cut_idx_1:]

            cut_idx = new_query.rfind("'''")
            new_query = new_query[:cut_idx]
            cut_idx = new_query.rfind("[YOU (AI assistant)]")
            if cut_idx == -1:
                print("Wrong prompt template!")
                raise AssertionError
            new_query_user = new_query[:cut_idx].strip()
            new_query_AI = new_query[cut_idx:].strip()
            if "private" in root:
                new_query_user = new_query_user.replace("from decision_company import read_csv_file,", "from decision_company import " + private_func).replace("# please import the necessary private functions from decision_company first", "")
                new_query_user = new_query_user + "\nAnalyze and try to use the provided custom function library named 'decision_company', which is enclosed within the BEGIN and END markers.\n\n--- decision_company BEGIN: ---\n\n" + function_add + "\n\n--- decision_company END ---\n\n"

            cut_idx_2 = dial_hist.rfind("[USER (data scientist)]")
            last_turn = dial_hist[cut_idx_2:]
            cut_idx_3 = last_turn.find("'''")
            python_code = last_turn[cut_idx_3:]
            cut_idx_4 = last_turn.rfind("[YOU (AI assistant)]")
            user_query = last_turn[:cut_idx_4]

            cut_idx = llm_resp.rfind("<pseudocode>")
            if cut_idx != -1:
                llm_resp = llm_resp[cut_idx:].replace("<pseudocode>", "")
            else:
                cut_idx = llm_resp.find("//")
                if cut_idx != -1:
                    llm_resp = llm_resp[cut_idx:]

            cut_idx = llm_resp.rfind("</pseudocode>")
            if cut_idx != -1:
                llm_resp = llm_resp[:cut_idx]

            private_AIR_02 = prompt_AIR_02_curr.format(head_info=head_info,dialogue_hist=dial_hist, user_query=user_query, pseudo_code=llm_resp, python_code=python_code, new_query_user=new_query_user, new_query_AI=new_query_AI)
            private_AIR_02 = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', private_AIR_02)

            message_obj = [{"role": self.role, "content": private_AIR_02}]

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            message_llm_pseudo = llm_response["choices"][0]["message"]["content"]

            llm_response_dict_curr[root] = message_llm_pseudo
            prompt_dict_curr[root] = private_AIR_02

            if count_save % 10 == 0:
                with open(save_dir, "w") as f_out:
                    json.dump(llm_response_dict_curr, f_out, indent = 4)

                with open(save_prompt, "w") as f_out:
                    json.dump(prompt_dict_curr, f_out, indent = 4)

        with open(save_dir, "w") as f_out:
            json.dump(llm_response_dict_curr, f_out, indent = 4)

        with open(save_prompt, "w") as f_out:
            json.dump(prompt_dict_curr, f_out, indent = 4)

        
    def multi_choice_base(self, condition):
        mc_prefix = "multi_choice_baseline"
        save_dir = os.path.join(self.output_path, 'multi_choice/01_' + mc_prefix + condition + "_" + self.engine_name + '.json')
        save_dir_01 = os.path.join(self.output_path, 'multi_choice/02_' + mc_prefix + condition + "_" + self.engine_name + '.json')
        llm_response_dict = {}
        llm_response_dict_01 = {}
        save_cnt = 0
        try:
            with open(save_dir, "r") as f_out:
                llm_response_dict = json.load(f_out)
        except:
            pass

        try:
            with open(save_dir_01, "r") as f_out:
                llm_response_dict_01 = json.load(f_out)
        except:
            pass

        for root in tqdm(self.data_dirs, desc="Processing roots"):
            if root in llm_response_dict and llm_response_dict[root] != "Failed!":
                continue

            ### Step 01: Generate code for the multi-choice question for only once ###
            save_cnt += 1
            with open(os.path.join(root, "reference/ref_code_hist.py"), 'r') as f_r:  
                hist_code = f_r.read() 
                hist_code = hist_code.replace("pd.read_csv(os.path.join(sys.argv[1], ", "pd.read_csv(")
                hist_code = hist_code.replace(".csv'))", ".csv')")
                hist_code = hist_code.replace("sys.path.append(sys.argv[1])", "")

            # The standard input format of GPTs: list of dicts which stores dialouge history
            with open(os.path.join(root, "reference/prompt_code_hist.json"), 'r') as f_j: 
                prompt_hist = json.load(f_j)
            
            prompt_hist[-1]["content"] = prompt_hist[-1]["content"] + " I will generate code between <code>...</code> below, which can assist me to answer the question in this step.\n<code>"
            message_obj = prompt_hist

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            message_llm = llm_response["choices"][0]["message"]["content"]

            ### Step 02: Execute the code generated from Step 01, and make final choice ###
            llm_response_dict_01[root] = message_llm
            idx_1 = message_llm.find("<code>")
            if idx_1 == -1:
                idx_1 = message_llm.find("'''")
                if idx_1 == -1:
                    idx_1 = message_llm.find("```python")
                    
            idx_2 = message_llm.find("</code>")
            if idx_2 == -1:
                idx_2 = message_llm.rfind("'''")
                if idx_2 == -1:
                    idx_2 = message_llm.rfind("```")
                    
            if idx_1 != -1 and idx_2 != -1:
                code_gen = message_llm[idx_1:idx_2].replace("<code>", "").replace("'''", "").replace("```python", "")
            else:
                code_gen = message_llm[:idx_2]

            hist_code = remove_pickle_code(hist_code)
            code_pred = hist_code + "\n" + code_gen
            code_pred = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', code_pred)
            code_pred = code_pred.replace("pd.read_csv(os.path.join(sys.argv[1], ", "pd.read_csv(")
            code_pred = code_pred.replace(".csv'))", ".csv')")
            code_pred = code_pred.replace("sys.path.append(sys.argv[1])", "")
            code_pred = format_code(code_pred, self.data_path)

            with capture_output() as (out, err):
                try:
                    exec(code_pred)  
                except Exception as e:
                    print(f"An error occurred: {e}", file=err)

            stdout, stderr = out.getvalue(), err.getvalue()
            processed_output = process_python_output(stdout)
            
            if "An error occurred:" in stderr:  
                print("Standard Error:", root)  
                print(stderr) 
                processed_output = stderr

            ai_dict = prompt_hist[-1]
            ai_dict["content"] = "<choice>"
            user_dict = prompt_hist[-2]
            trigger_sent = "Please generate the python code (with pandas version 2.0.3 and matplotlib version 3.7.4)"
            idx_cut = user_dict["content"].rfind(trigger_sent)
            if idx_cut != -1:
                user_dict["content"] = user_dict["content"][:idx_cut]
                
            new_prompt = "--- Code Generated: ---\n'''" + code_gen + "'''\n\n--- Code Execution Result: ---\n" + processed_output + "\n\nBased on the code generated and its execution results for the first question of this turn, choose the most appropriate option to answer the latter question and directly provide the choice between <choice>...</choice>."
            user_dict["content"] = user_dict["content"] + new_prompt

            prompt_hist[-2] = user_dict
            prompt_hist[-1] = ai_dict
            message_obj = prompt_hist

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine, max_tokens=500)
            message_llm = llm_response["choices"][0]["message"]["content"]
            llm_response_dict[root] = message_llm

            if save_cnt % 2 == 0:
                with open(save_dir, "w") as f_out:
                    json.dump(llm_response_dict, f_out, indent = 4)

                with open(save_dir_01, "w") as f_out:
                    json.dump(llm_response_dict_01, f_out, indent = 4)

        with open(save_dir, "w") as f_out:
            json.dump(llm_response_dict, f_out, indent = 4)

        with open(save_dir_01, "w") as f_out:
            json.dump(llm_response_dict_01, f_out, indent = 4)


    def multi_choice_AIR(self, condition):
        llm_response_dict_AIR_pseudo = {}
        save_cnt = 0
        mc_prefix = "multi_choice_" + self.model_version
        save_dir =  os.path.join(self.output_path, 'multi_choice/01_AIR_' + mc_prefix + condition + "_" + self.engine_name + '.json')
        try:
            with open(save_dir, "r") as f_out:
                llm_response_dict_AIR_pseudo = json.load(f_out)
        except:
            pass

        for root in tqdm(self.data_dirs, desc="Processing Pseudocodes"):
            if root in llm_response_dict_AIR_pseudo and llm_response_dict_AIR_pseudo[root] != "Failed!":
                continue

            with open(os.path.join(root, "reference/prompt_code_hist.json"), 'r') as f_j: 
                prompt_AIR = json.load(f_j)
                try:
                    dial_hist_student = prompt_AIR[:-1]
                    dial_hist_teacher = prompt_AIR[:-2]
                    teacher_query = dial_hist_teacher[-2]["content"]
                    if dial_hist_teacher[-2]["role"] != "user":
                        print("Wrong Interaction History!")
                        print(root)
                    if dial_hist_teacher[-1]["role"] != "assistant":
                        print("Wrong Interaction History!")
                        print(root)
                except IndexError:
                    print(root)
                    continue

            save_cnt += 1
            teacher_user = {"role": "user"}
            teacher_user["content"] = prompt_AIR_teacher_user
            teacher_AI = {"role": "assistant"}
            teacher_AI["content"] = prompt_AIR_teacher_AI
            dial_hist_teacher.append(teacher_user)
            dial_hist_teacher.append(teacher_AI)

            message_obj = dial_hist_teacher

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            message_llm = llm_response["choices"][0]["message"]["content"]

            student_AI = {"role": "assistant"}
            student_AI["content"] = prompt_AIR_student_AI
            cut_idx = message_llm.find("<pseudocode>")
            if cut_idx != -1:
                message_llm = message_llm[cut_idx:].replace("<pseudocode>", "")

            cut_idx = message_llm.find("</pseudocode>")
            if cut_idx != -1:
                message_llm = message_llm[:cut_idx]
            dial_hist_student[-1]["content"] = prompt_AIR_student_user.format(dialogue_hist=dial_hist_student[-1]["content"], user_query=teacher_query, pseudo_code=message_llm)
            dial_hist_student.append(student_AI)
            message_obj = dial_hist_teacher

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine)
            message_llm_pseudo = llm_response["choices"][0]["message"]["content"]
            cut_idx = message_llm_pseudo.find("<pseudocode>")
            if cut_idx != -1:
                message_llm_pseudo = message_llm_pseudo[cut_idx:].replace("<pseudocode>", "")
            cut_idx = message_llm_pseudo.find("</pseudocode>")
            if cut_idx != -1:
                message_llm_pseudo = message_llm_pseudo[:cut_idx]

            llm_response_dict_AIR_pseudo[root] = message_llm_pseudo

            if save_cnt % 2 == 0:
                with open(save_dir, "w") as f_out:
                    json.dump(llm_response_dict_AIR_pseudo, f_out, indent = 4)

        with open(save_dir, "w") as f_out:
            json.dump(llm_response_dict_AIR_pseudo, f_out, indent = 4)

        return save_dir


    def multi_choice_agents(self, condition, max_turn):
        if self.model_version == "inter_agent":
            save_dir_AIR = self.multi_choice_AIR(condition)
            with open(save_dir_AIR, "r") as f_out:
                llm_response_dict_AIR_pseudo = json.load(f_out)

        mc_prefix = "multi_choice_" + self.model_version
        save_dir =  os.path.join(self.output_path, 'multi_choice/' + mc_prefix + condition + "_" + self.engine_name + '.json')

        llm_response_dict = {}
        save_cnt = 0
        try:
            with open(save_dir, "r") as f_out:
                llm_response_dict = json.load(f_out)
        except:
            pass

        for root in tqdm(self.data_dirs, desc="Processing roots"):
            if root in llm_response_dict and llm_response_dict[root] != "Failed!":
                continue

            save_cnt += 1
            if self.model_version == "inter_agent":
                try:
                    mc_pseudo_AIR = llm_response_dict_AIR_pseudo[root]
                except KeyError:
                    continue

            with open(os.path.join(root, "reference/ref_code_hist.py"), 'r') as f_r:  
                hist_code = f_r.read() 
                hist_code = hist_code.replace("pd.read_csv(os.path.join(sys.argv[1], ", "pd.read_csv(")
                hist_code = hist_code.replace(".csv'))", ".csv')")
                hist_code = hist_code.replace("sys.path.append(sys.argv[1])", "")

            with open(os.path.join(root, "reference/prompt_code_hist.txt"), 'r') as f_w:  
                prompt_hist = f_w.read() 

            dialogue_idx1 = prompt_hist.rfind("Interactions begin:")
            dialogue_idx2 = prompt_hist.rfind('Please choose the best option and directly provide the choice between <choice>...</choice>.')
            if dialogue_idx2 == -1:
                dialogue_idx2 = prompt_hist.rfind("[YOU (AI assistant)]")
            dialogue_hist = prompt_hist[dialogue_idx1:dialogue_idx2].replace("Interactions begin:", "")

            dialogue_hist = remove_pickle_code(dialogue_hist)

            if "credit" in root:
                df_add = credit_df
                if self.model_version == "inter_agent":
                    prompt_react_temp = prompt_react_code_hist_analysis_credit_AIR
                else:
                    prompt_react_temp = prompt_react_code_hist_analysis_credit  
            if "atp_tennis" in root:
                df_add = atp_df
                if self.model_version == "inter_agent":
                    prompt_react_temp = prompt_react_code_hist_analysis_ATP_AIR
                else:
                    prompt_react_temp = prompt_react_code_hist_analysis_ATP  
            if "fastfood" in root:
                df_add = fastfood_df
                if self.model_version == "inter_agent":
                    prompt_react_temp = prompt_react_code_hist_analysis_fastfood_AIR
                else:
                    prompt_react_temp = prompt_react_code_hist_analysis_fastfood  
            if "melb_" in root:
                df_add = melb_df
                if self.model_version == "inter_agent":
                    prompt_react_temp = prompt_react_code_hist_analysis_melb_AIR
                else:
                    prompt_react_temp = prompt_react_code_hist_analysis_melb  
            if "laptop" in root:
                df_add = laptop_df
                if self.model_version == "inter_agent":
                    prompt_react_temp = prompt_react_code_hist_analysis_laptop_AIR
                else:
                    prompt_react_temp = prompt_react_code_hist_analysis_laptop  
            
            if self.model_version == "inter_agent":
                react_prompt = prompt_react_temp.format(dialogue_hist=dialogue_hist, pseudo_code=mc_pseudo_AIR)
            else:
                react_prompt = prompt_react_temp.format(dialogue_hist=dialogue_hist)
            react_prompt = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', react_prompt)

            out_fn = "prompt_" + self.model_version + "_analysis_turn_1.txt"
            with open(os.path.join(root, "reference/" + out_fn), 'w') as f_out:  
                f_out.write(react_prompt) 
            
            message_obj = [{"role": self.role, "content": react_prompt}]

            '''For GPT family only'''
            llm_response = get_llm_response(message_obj, engine=self.llm_engine, max_tokens=500)
            message_llm = llm_response["choices"][0]["message"]["content"]

            code_histo = ""
            for i in range(max_turn):
                idx_1 = message_llm.find("'''")
                idx_2 = message_llm.rfind("'''")
                if idx_1 == idx_2:
                    code_gen = message_llm[:idx_2].replace("'''", "")
                else:
                    code_gen = message_llm[idx_1:idx_2].replace("'''", "")

                code_histo = remove_pickle_code(code_histo)
                code_histo = code_histo + code_gen + "\n\n"
                hist_code = remove_pickle_code(hist_code)
                
                code_pred = hist_code + "\n" + code_histo
                code_pred = code_pred.replace("target_customer_segments = [1, 2]", "")
                code_pred = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', code_pred)
                code_pred = format_code(code_pred, self.data_path)

                with capture_output() as (out, err):
                    try:
                        exec(code_pred)  
                    except Exception as e:
                        print(f"An error occurred: {e}", file=err)

                stdout, stderr = out.getvalue(), err.getvalue()
                processed_output = process_python_output(stdout)

                if "An error occurred:" in stderr:  
                    print("Standard Error:", root)  
                    print(stderr) 
                    processed_output = stderr

                turn_left = max_turn - i

                last_template = find_last_template(react_prompt, "################### Your Answer Starts Here: ###################")
                if i == 0:
                    trigger_str = "Code 1:"
                    result_str = "Result 1:\n" + df_add
                    turn_num = "\n\nTurn " + str(i+2) +":"
                    turn_left_msg = turn_num + "\n# " + str(turn_left) + " turns left to provide final answer. Please only generate a small step in 'Thought' which is different from the Example above, a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. \nThought 2:"
                else:
                    trigger_str = "Thought " + str(i+1) +":"
                    result_str = "Result "+ str(i+1) +":" + "\n" + processed_output + "\n"
                    turn_num = "\n\nTurn " + str(i+2) +":"
                    turn_left_msg = turn_num + "\n# " + str(turn_left) + " turns left to provide final answer. Please only generate a small step in 'Thought' which is different from the Example above, a code segment in 'Code' (with proper print) and 'Act' in this turn, no need to generate 'Result'. \nThought " + str(i+2) +":"
                    if turn_left == 1:
                        turn_left_msg = "\n\n# You have to provide final answer in this turn! Please only 'Act' and 'Answer' in this turn, no need to generate Result.\nThought " + str(i+2) +": Analyze all results I have.\nAct:"

                last_template_update = last_template.replace(trigger_str, trigger_str + "\n" + message_llm + "\n" + result_str + turn_left_msg)

                react_prompt = react_prompt.replace(last_template, last_template_update)
                react_prompt = re.sub(r'(?:[ \t]*(?:\r?\n)){3,}', '\n\n', react_prompt)

                out_fn = "prompt_" + self.model_version + "_analysis_turn_" + str(i+2) + ".txt"
                with open(os.path.join(root, "reference/" + out_fn), 'w') as f_out:  
                    f_out.write(react_prompt) 

                message_obj = [{"role": self.role, "content": react_prompt}]

                '''For GPT family only'''
                llm_response = get_llm_response(message_obj, engine=self.llm_engine, max_tokens=500)
                message_llm = llm_response["choices"][0]["message"]["content"]

                if "Terminate" in message_llm:
                    break

            llm_response_dict[root] = message_llm

            if save_cnt % 2 == 0:
                with open(save_dir, "w") as f_out:
                    json.dump(llm_response_dict, f_out, indent = 4)

        with open(save_dir, "w") as f_out:
            json.dump(llm_response_dict, f_out, indent = 4)
