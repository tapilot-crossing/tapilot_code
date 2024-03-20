## Eval Instruction

You can find a more detailed instruction about eval here.

### Step 1: Evaluate Each Settings

  To quantitatively evaluate the performance of different models based on the results obtained from the prediction code execution. Utilizing specific `eval.py` within each dialogue turn's folder, this step facilitates a comparative evaluation of the results.
  
  - For code generation setting, you will need to run `eval_code_gen.py` by using:

    ```bash
    sh ./run/eval_code_gen.sh
    ```

  - For analysis setting, you will need to run `eval_multi_choice.py` by using:

    ```bash
    sh ./run/eval_multi_choice.sh
    ```
  
  This step generates `eval_stats.json`, which contains detailed performance evaluation located under each dialouge turn folder.

### Step 2: Generate Overall Performance Report

  To compile and summarize the performance metrics across all settings, providing a comprehensive overview of the LLM agents' capabilities.

  - For the overall performance report generation, you will need to run `eval_one_click.py` by using:

    ```bash
    sh ./run/eval_overall.sh
    ```

  You will get the overall performance report on your model.
