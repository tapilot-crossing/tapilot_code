## Classification of Different Shell Files

### Category 1: Generate LLM responses 

  You can use `run_code_gen.sh` and `run_multi_choice.sh` to generate LLM response. As for running different **actions** with `run_clarification_preparation.sh` and `run_plotqa.sh`, please refer to the README in methods folder.

### Category 2: Postprocessing 

  You can use `postprocess_code_gen.sh` to extract valid python code in LLM responses. 

### Category 3: Eval Preparation

 First, you need to generate results from predicted and reference code for each data, using `run_reference_code.sh` and `run_prediction_code.sh` respectively.

 NOTE: During postprocessing step, the directory path for reading CSV files should be already correctly specified. For instance, `pd.read_csv('atp_tennis.csv')` should be modified to include the full path, as below:

  ```bash
  pd.read_csv(os.path.join(sys.argv[1], 'atp_tennis.csv'))
  ```

### Category 4: Evaluation

 To quantitatively evaluate the performance of different models based on the results obtained from the prediction code execution. Utilizing specific `eval.py` within each dialogue turn's folder, this step facilitates a comparative evaluation of the results. You can use `eval_code_gen.sh` and `eval_multi_choice.sh` respectively. This step generates `eval_stats.json`, which contains detailed performance evaluation located under each dialouge turn folder.

 To compile and summarize the performance metrics across all settings, providing a comprehensive overview of the LLM agents' capabilities. You can use `eval_overall.sh`.
