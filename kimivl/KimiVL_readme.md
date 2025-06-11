# Steps to reproduce the results using Kimi-VL

1. Install the required dependencies:
```bash
pip install transformers torch pillow
```

2. Navigate to the kimivl directory:
```bash
cd kimivl
```

3. Run the evaluation script:
```bash
python kimivl_3d_test.py \
    --question_file /path/to/questions.json \
    --answer_file /path/to/answers.json \
    --image_folder /path/to/images \
    --export_json results.json \
    --model_path moonshotai/Kimi-VL-A3B-Thinking \
    --device cuda:0
```

4. Convert the generated results to the format used for evaluation:
```bash
python convert_json_files.py
```

5. Run the evaluation script:
```bash
cd ..
bash scripts/eval/sqa3d_distributed.sh
```

# Notes:
- The script uses the Kimi-VL model from Hugging Face (moonshotai/Kimi-VL-A3B-Thinking)
- Make sure you have sufficient GPU memory for running the model
- The script processes images in batches to manage memory usage
- You may need to adjust the batch size and other parameters based on your GPU memory
- The model's output format may need to be adjusted based on the actual response format 