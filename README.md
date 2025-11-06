# MetaLint
Code for Non-Idiomatic Python & Java Code Detection with LLMs (work done with Oracle Labs East)

## Training & Evaluation Workflow

### SFT training
First train SFT (alignment-handbook):

```bash
cd alignment-handbook

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py /path/to/sft/config.yaml
``` 

### SFT Evaluation

Then evaluate SFT checkpoints at various steps to find the best one:

```bash
cd ..

python src/model/eval_llm_meta_linter_vllm_futures.py --model_name /path/to/checkpoint --write_path /path/to/where/you/want/to/write/results --test_file data/ruff_meta_linting/test_v4_new_format_with_lineno.json
```

To compute metrics (transfer setting):

```bash
python src/metrics/meta_linting/idiom_detection_and_localization_v3.py <STEP_NO>
```

```bash
python src/metrics/meta_linting/idiom_detection_and_localization_all_idioms.py <STEP_NO>
```

To launch vLLM server (needed for both evaluation and DPO data generation):

```bash
bash src/model/launch_vllm_server.sh /path/to/model/checkpoint
```

### DPO data generation

Then based on the best checkpoint path generate samples for DPO training:

```bash
python src/model/eval_llm_meta_linter_vllm_multi_sample.py --model_name /path/to/best/checkpoint --write_path data/dpo_samples/<FILENAME>
```

convert saved DPO samples into training pairs in Huggingface format:

```bash
python src/dpo/convert_dpo_samples_to_pairs.py
```

After this upload the data to Huggingface!

### DPO training

```bash
cd alignment-handbook

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py /path/to/dpo/config.yaml
``` 

## Citation

If you find our work useful, please cite us as follows:
```
@article{naik2025metalint,
  title={MetaLint: Generalizable Idiomatic Code Quality Analysis through Instruction-Following and Easy-to-Hard Generalization},
  author={Naik, Atharva and Baghel, Lawanya and Govindarajan, Dhakshin and Agrawal, Darsh and Fried, Daniel and Rose, Carolyn},
  journal={arXiv preprint arXiv:2507.11687},
  year={2025}
}
```
