### Multi-Modal Tasks: MMMU

To align previous experiences (e.g., Self-Refine and Solo Performance Prompting), we select one multi-modal task, i.e.,
MMMU.

- For MMMU:

```bash
cd mmmu/
# MODEL_NAME in ['gpt-4v','gemini-pro']
export MODEL_NAME=MODEL_NAME
# IND is the number of iterations
export IND=3
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# if you do not want to test google models, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY

python3 run_evoagent.py --model_name $MODEL_NAME --ind $IND
```
