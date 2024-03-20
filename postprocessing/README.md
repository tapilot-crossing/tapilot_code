## Postprocessing Instruction

The output format from a large language model (LLM) can vary significantly depending on the prompt used. To address this, we introduce a rule-based method for extracting Python code. It's important to note, though, that you may need to substantially adapt the postprocessing steps to fit the specifics of your own methods and evaluations. If you encounter unexpected outputs from certain LLMs, we suggest utilizing GPT-3.5 as a more reliable source for code extraction.

To apply our postprocessing method, execute the following command:

```bash
sh ./run/postprocess_code_gen.sh
```
During postprocessing, the directory path for reading CSV files is correctly specified. For instance, `pd.read_csv('atp_tennis.csv')` will be modified to include the full path, as `pd.read_csv(os.path.join(sys.argv[1], 'atp_tennis.csv'))`.
