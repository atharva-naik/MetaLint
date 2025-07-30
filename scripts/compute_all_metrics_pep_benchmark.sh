TEST_FILE="./data/pep_benchmark/test_pep_v2.json"

for PREDS_PATH in ./data/pep_benchmark_preds_v2/*; do
  [ -f "$PREDS_PATH" ] || continue  # skip if not a regular file
  [[ "$PREDS_PATH" == *.jsonl ]] || continue  # skip if not .jsonl
  echo "Computing metrics for: $PREDS_PATH"
  python src/metrics/meta_linting/idiom_detection_and_localization_v4.py -p $PREDS_PATH -tf $TEST_FILE -pep
done



# python src/metrics/meta_linting/idiom_detection_and_localization_v4.py -p "data/pep_benchmark_preds_v2/deepseek_r1_distill_qwen_7b_untrained_think_preds.jsonl" -tf $TEST_FILE -pep

# python src/metrics/meta_linting/idiom_detection_and_localization_v4.py -p "data/pep_benchmark_preds_v2/deepseek_r1_distill_qwen_14b_untrained_think_preds.jsonl" -tf $TEST_FILE -pep

# python src/metrics/meta_linting/idiom_detection_and_localization_v4.py -p "data/pep_benchmark_preds_v2/deepseek_r1_distill_qwen_32b_untrained_think_preds.jsonl" -tf $TEST_FILE -pep

# python src/metrics/meta_linting/idiom_detection_and_localization_v4.py -p "data/pep_benchmark_preds_v2/gpt_4.1_preds.jsonl" -tf $TEST_FILE -pep

# python src/metrics/meta_linting/idiom_detection_and_localization_v4.py -p "data/pep_benchmark_preds_v2/gpt_4o_preds.jsonl" -tf $TEST_FILE -pep
