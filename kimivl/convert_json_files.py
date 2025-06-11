import json
import argparse

def convert_results(input_file: str, output_file: str):
    """
    Convert the results from Kimi-VL format to the evaluation format.
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    """
    # Read the input file
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Convert to evaluation format
    eval_results = []
    for result in results:
        eval_result = {
            "question_id": result["question_id"],
            "video": result["video"],
            "question": result["question"],
            "answer": result["answer"],
            "type": result["type"]
        }
        eval_results.append(eval_result)
    
    # Save the converted results
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    convert_results(args.input, args.output) 