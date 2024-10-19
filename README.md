
<h1 align="center">EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms</h1>
<p align="center">
[<a href="https://evo-agent.github.io/">Website</a>]
[<a href="https://arxiv.org/pdf/2406.14228">Paper</a>] 

</p>

## What's New
+  [2024.06.21] We release EvoAgent.
   + The website is available at [EvoAgent](https://evo-agent.github.io/).
   + The paper is available at [EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms](https://arxiv.org/abs/2406.14228).

## EvoAgent

EvoAgent is a generic method to automatically extend expert agents to multi-agent systems via the evolutionary algorithm. Specifically, to align human society, each agent can be considered as individuals that can procreate its population across successive generations. Motivated by this mechanism, we can simulate such a human behavior to automatically generate multiple agents based on any pre-defined agents.

<p align="center">
<img width="70%" alt="image" src="./assets/framework.png">    
</p>

More details are in the paper.

## Experiment

### NLP and Multi-Modal Tasks: SPP and MMMU
To align previous experiences (e.g., Self-Refine and Solo Performance Prompting), we select three NLP knowledge-intensive and reasoning-intensive tasks, i.e., Logic Grid Puzzle, Trivia Creative Writing and Codenames Collaborative, and one multi-modal task, i.e., MMMU.

#### Prerequisites

Create a conda environment and install dependency:
```bash
conda create -n spp python=3.9
conda activate spp
pip install -r requirements.txt
```

#### Code

- For NLP tasks:
```bash
cd spp/
# task in ['writing', 'logic', 'code']
export task=writing
# MODEL_NAME in ['gpt-4-X','gpt-3.5-turbo-X','llama-13b-chat']
export MODEL_NAME=MODEL_NAME
# DATA_TYPE in ['openai', 'azure', 'gemini', 'small']
export DATA_TYPE=openai
# IND is the number of iterations
export IND=3
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# if you do not want to test google models, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
# for Logic Grid Puzzle and Trivia Creative Writing
python3 llm_evoagent.py --model_name $MODEL_NAME --data_type $DATA_TYPE --method evoagent --ind $IND
# for Codenames Collaborative
python3 llm_evoagent_codenames.py --model_name $MODEL_NAME --data_type $DATA_TYPE --method evoagent --ind $IND
```
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
### Interactive Scientific Solving Simulation: ScienceWorld

We choose ScienceWorld, a complex interactive environment requiring skills in long-term memory, sub-task decomposition, and scientific and commonsense knowledge. Here, we evaluate 30 scientific tasks in ScienceWorld to demonstrate the capability of EvoAgent in solving tasks in more challenging open-world environments. 

This code is adapted from [AgentTuning](https://github.com/THUDM/AgentTuning/tree/main/eval_heldout/science-world) and [SwiftSage](https://github.com/yuchenlin/SwiftSage)

#### Prerequisites

Create a conda environment and install dependency:
```bash
cd scienceworld/
conda create -n sciworld python=3.8 pip
conda activate sciworld
pip3 install scienceworld==1.1.3
pip3 install -r requirements.txt
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install -c conda-forge openjdk # if needed 
```


#### Code

```bash
export OPENAI_API_KEY=YOUR_OPENAI_KEY
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

### TravelPlanner
We test EvoAgent on the sole-planning mode. The sole-planning mode ensures that no crucial information is missed, thereby enabling agents to focus on planning itself.
Please take a look at the paper for more details.

#### Prerequisites

1. Create a conda environment and install dependency:
```bash
cd travelplanner/
conda create -n travelplanner python=3.9
conda activate travelplanner
pip install -r requirements.txt
```
2. Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `TravelPlanner` directory (i.e., `your/path/TravelPlanner`).

#### Code

This code is adapted from [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner)

```bash
export OUTPUT_DIR=path/to/your/output/file
# We support MODEL in ['gpt-3.5-turbo-X','gpt-4-1106-preview','gemini','mistral-7B']
export MODEL_NAME=MODEL_NAME
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# if you do not want to test google models, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
# IND is the number of iterations
export IND=3
# GROUP_NUM is the population size generated in each iteration
export GROUP_NUM=3
# SELECT_STRATEGY in ['random', 'all', 'pk']
export SELECT_STRATEGY=all
export SET_TYPE=validation
# STRATEGY in ['direct','cot','react','evoagent','group'].
# Collaboration is our method in the main result. Group is our methods in the ablation studies.
export STRATEGY=evoagent

cd tools/planner
python sole_planning.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --ind $IND --group_num $GROUP_NUM --select_strategy $SELECT_STRATEGY
```



## Citation

If you find this work useful in your method, you can cite the paper as below:

```bash
@misc{yuan2024evoagent,
      title={EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms}, 
      author={Siyu Yuan and Kaitao Song and Jiangjie Chen and Xu Tan and Dongsheng Li and Deqing Yang},
      year={2024},
      eprint={2406.14228},
      archivePrefix={arXiv},
}
```
