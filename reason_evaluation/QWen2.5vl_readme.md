# Step to reproduce the results using QWen 2.5VL
1. Checkout to the branch "leihan_chen/qwen2.5vl" to get the code for the evaluation.
2. Please check the [QWen 2.5VL](https://github.com/QwenLM/Qwen2.5-VL) GitHub repository for the model and environment setup.
3. Open the evaluation folder.
```bash
cd reason_evaluation
```
4. Run the following python script.
```python
python3 qwen2.5vl_3d_test.py 
```
5. Convert generated results to the format used for evaluation script.
```python
python3 convert_json_files.py 
```
6. Run the evaluation script.
```bash
cd ..
bash scripts/eval/sqa3d_distributed.sh
```

# NOTE:
- The generated files from QWen can be problematic for JSON output format.
- Convert script cannot always conver all json entries due to the inconsistent format of the generated files by QWen.
- Please modify qwen2.5vl_3d_test.py file for enabling sglang or vllm inference with port setting to satisfy the serving condition of the model.
- There are still maximum token limitation of the input sequence which blocks the usage of all image sequences.
- Using following command to launch a SGLang server.
```bash
CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --port 30000 --chat-template qwen2-vl --tp 2 --mem-fraction-static 0.7
```
