## Action Setting Instruction

#### Code Generation Actions

There are two types of action in code gen setting: `correction` and `clarification`.

- To run `clarification` action data, you will need to firstly run `tapilot_clarification_preparation.py` with:
   
    ```bash
    sh ./run/run_clarification_preparation.sh
    ```
    
- Then you can run either the `correction` or `clarification` action data as normal ones with proper settings using:
   
    ```bash
    sh ./run/run_code_gen.sh
    ```   

#### Multi-choice Actions

There are four types of action in code gen setting: `unanswerable` and `best_guess` and `plotQA` and `analysis`.

- To run `plotQA` action data, you can run `tapilot_plotqa.py` with:
   
    ```bash
    sh ./run/run_plotqa.sh
    ```
    
- To run other kinds of Multi-choice actions, you can treat them as normal ones with proper settings using:
   
    ```bash
    sh ./run/run_multi_choice.sh
    ```

    NOTE: The **max_turn** in ReAct and AIR setting can be different in different actions. For example, when running `analysis` setting, can set **max_turn = 6**; and set **max_turn = 4** for others.
