export task=writing
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_API_BASE="your_openai_base"
export OPENAI_API_VERSION="your_openai_version"
export GOOGLE_API_KEY="your_gemini_api_key"

python3 llm_collaborate.py --model_name $MODEL_NAME --data_type $DATA_TYPE --method collaborate --ind $IND

python3 llm_collaborate_codenames.py --model_name $MODEL_NAME --data_type $DATA_TYPE --method collaborate --ind $IND