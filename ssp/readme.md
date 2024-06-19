### NLP Tasks: SSP
To align previous experiences (e.g., Self-Refine and Solo Performance Prompting), we select three NLP knowledge-intensive and reasoning-intensive tasks, i.e., Logic Grid Puzzle, Trivia Creative Writing and Codenames Collaborative.
- For NLP tasks:
```bash
cd ssp/
# task in ['writing', 'logic', 'code']
export task=writing
# MODEL_NAME in ['gpt-4-turbo-X','gpt-3.5-turbo-X','llama-13b-chat']
export MODEL_NAME=MODEL_NAME
# DATA_TYPE in ['openai', 'azure', 'gemini', 'small']
export DATA_TYPE=openai
# IND is the number of iterations
export IND=3
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# if you do not want to test google models, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
# for Logic Grid Puzzle and Trivia Creative Writing
python3 llm_collaborate.py --model_name $MODEL_NAME --data_type $DATA_TYPE --method collaborate --ind $IND
# for Codenames Collaborative
python3 llm_collaborate_codenames.py --model_name $MODEL_NAME --data_type $DATA_TYPE --method collaborate --ind $IND
```
