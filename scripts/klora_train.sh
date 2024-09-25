PYTHON_VERSION="3.11"
MODEL_NAME="ise-uiuc/Magicoder-S-DS-6.7B"
python -m src.knowledge_adapters.train_peft --train_dataset ./data/Python-docs/whatsnew.jsonl --output_dir "experiments/${MODEL_NAME}-${PYTHON_VERSION}-klora" --python_version "${PYTHON_VERSION}" --model_name $MODEL_NAME