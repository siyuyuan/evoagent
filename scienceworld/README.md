### Interactive Scientific Solving Simulation: ScienceWorld

We choose ScienceWorld, a complex interactive environment requiring skills in long-term memory, sub-task decomposition, and scientific and commonsense knowledge. Here, we evaluate 30 scientific tasks in ScienceWorld to demonstrate the capability of EvoAgent in solving tasks in more challenging open-world environments. 

This evaluation code is adapted from [AgentTuning](https://github.com/THUDM/AgentTuning/tree/main/eval_heldout/science-world) and [SwiftSage](https://github.com/yuchenlin/SwiftSage)

```bash
cd scienceworld/
export OPENAI_API_KEY=sk-your-openai-api-key
# MODEL_NAME in ['gpt-4-X','gpt-3.5-turbo-X']
export MODEL_NAME=MODEL_NAME

for task in {0..29}
do
    python eval_evoagent.py \
        --task_nums $task \
        --output_path logs/$MODEL_NAME \
        --model_name $MODEL_NAME
done
```
