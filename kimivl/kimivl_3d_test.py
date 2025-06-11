import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from typing import List, Dict, Any

import pdb

def load_images(folder_path: str):
    """
    Load images from a folder.
    Args:
        folder_path (str): Path to the folder containing images.
    Returns:
        list: List of loaded images.
    """
    regular_images = []
    depth_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            regular_images.append(img_path)
        elif filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            depth_images.append(img_path)
    return regular_images, depth_images

def process_rgb_images(images: list):
    """
    Process RGB images.
    Args:
        images (list): List of RGB images.
    Returns:
        list: List of processed RGB images.
    """
    return [Image.open(path).convert('RGB') for path in images]

def process_depth_image(depth_image, min_depth=None, max_depth=None):
    """
    Convert depth image to RGB using a colormap.
    Args:
        depth_image: PIL Image or numpy array of depth values
        min_depth: minimum depth value for normalization
        max_depth: maximum depth value for normalization
    Returns:
        PIL Image in RGB format
    """
    import cv2
    import numpy as np
    
    # Convert to numpy if PIL Image
    if isinstance(depth_image, Image.Image):
        depth_array = np.array(depth_image)
    else:
        depth_array = depth_image
        
    # Normalize depth values
    if min_depth is None:
        min_depth = np.min(depth_array[depth_array > 0])
    if max_depth is None:
        max_depth = np.max(depth_array)
    
    # Normalize to 0-1 range
    depth_normalized = (depth_array - min_depth) / (max_depth - min_depth)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # Apply colormap (using jet colormap)
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(depth_colored)

def process_depth_images(depth_images: list):
    """
    Process depth images.
    Args:
        images (list): List of depth images.
    Returns:
        list: List of processed depth images.
    """
    return [process_depth_image(Image.open(path)) for path in depth_images]

def load_questions_json(file_path: str):
    """
    Load questions from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        list: List of questions.
    """
    with open(file_path, "r") as f:
        questions = json.load(f)
    return questions

def add_answers_to_questions(questions: list, answer_file: str):
    """
    Add answers to questions by matching question_id.
    Args:
        questions (list): List of questions.
        answer_file (str): Path to the answer JSON file.
    Returns:
        list: List of questions with answers.
    """
    with open(answer_file, "r") as f:
        answers = json.load(f)
    # Build a lookup dict for answers by question_id
    answer_lookup = {a["question_id"]: a for a in answers}
    for question in questions:
        qid = question["question_id"]
        if qid in answer_lookup:
            question["answer"] = answer_lookup[qid]["text"]
            question["type"] = answer_lookup[qid]["type"]
        else:
            question["answer"] = None
            question["type"] = None
    return questions

def find_image_paths(questions: list, folder_path: str, sample_rate: int = 1):
    """
    Find image paths in questions.
    Args:
        questions (list): List of questions.
        folder_path (str): Path to the folder containing images.
        sample_rate (int): Sample rate for images.
    Returns:
        list: questions List with image paths.
    """
    image_paths = []
    for question in questions:
        scene_name = question["video"]
        scene_folder_path = os.path.join(
            folder_path, scene_name, scene_name + "_sens", "color"
        )
        count = 0
        for filename in os.listdir(scene_folder_path):
            if filename.endswith(".jpg"):
                count += 1
                if count % sample_rate == 0:
                    img_path = os.path.join(scene_folder_path, filename)
                    image_paths.append(img_path)
        question["scene_images_path"] = image_paths
        image_paths = []  # Reset image_paths for the next question
    return questions


def get_data(image_folder_path: str, scene: str, data_type: str = "rgb", sample_rate: int = 5) -> list:
    """
    Get image data from a scene.
    Args:
        image_folder_path (str): Path to the folder containing images.
        scene (str): Scene name.
        depths (bool): Whether to include depths.
        sample_rate (int): Sample rate for images.
    Returns:
        list: A tensor of images, a tensor of depths
    """
    if data_type == "rgb":
        style = {"dir": "color", "ext": ".jpg"}
    elif data_type == "depth":
        style = {"dir": "depth", "ext": ".png"}
    elif data_type == "pose":
        style = {"dir": "pose", "ext": ".txt"}
    else:
        raise ValueError(f"Invalid image type: {data_type}")
        
    data_dir = os.path.join(image_folder_path, scene, scene + "_sens", style["dir"])
    data = [i for i in os.listdir(data_dir) if i.endswith(style["ext"])][::sample_rate]
    assert [int(i.split(".")[0]) for i in data] == list(range(0, len(data) * sample_rate, sample_rate)), "Images are not in order"
    data = [os.path.join(data_dir, image) for image in data]
    return data


def kimivl_video_test(
    model, processor, image_paths: list, text_prompt: str, model_path: str, device: str = "cuda:0"
):
    """
    Run inference with Kimi-VL model on a sequence of images.
    Args:
        image_paths (list): List of paths to images
        text_prompt (str): Text prompt for the model
        model_path (str): Path to the model
        device (str): Device to run inference on
    Returns:
        str: Model's response
    """
    # Load images as PIL
    images = process_rgb_images(image_paths)
    print("Images loaded")

    # Prepare messages in the correct format for multimodal models
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img} for img in images
            ] + [{"type": "text", "text": text_prompt}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "You are a help assistant to answer the question concisely with your separate reasoning trace.",
                }
            ],
        }
    ]
    print("Messages prepared")
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    print("Text prepared")
    inputs = processor(
        images=images,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    print("Inputs prepared")
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,  # Shorter for SQA3D answers
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    gc.collect()
    torch.cuda.empty_cache()
    return response

def save_output(output_text: str, question: dict, output_file_path: str):
    """
    Save the model's output to a file.
    Args:
        output_text (str): Model's output text
        question (dict): Question dictionary
        output_file_path (str): Path to save the output
    """
    # Extract the answer from the output text
    # This is a simple implementation - you may need to adjust based on actual output format
    answer = output_text.strip()
    
    # Create output dictionary
    output_dict = {
        "question_id": question.get("question_id", ""),
        "video": question.get("video", ""),
        "question": question.get("question", ""),
        "answer": answer,
        "type": question.get("type", "")
    }
    
    # Save to file
    with open(output_file_path, "a") as f:
        json.dump(output_dict, f)
        f.write("\n")

def main(
    question_file_path: str,
    answer_file_path: str,
    image_folder_path: str,
    export_json_path: str,
    model_path: str,
    device: str = "cuda:0",
    sample_rate: int = 5
):
    """
    Main function to run the evaluation.
    Args:
        question_file_path (str): Path to questions JSON file
        answer_file_path (str): Path to answers JSON file
        image_folder_path (str): Path to images folder
        export_json_path (str): Path to export results
        model_path (str): Path to model
        device (str): Device to run inference on
    """
    
    if device != "cuda":
        torch.cuda.set_device(device)  # Force CUDA to use the specified device, i.e. "cuda:0" or "cuda:1" etc.
    
    # Load questions and answers
    questions = load_questions_json(question_file_path)
    questions = add_answers_to_questions(questions, answer_file_path)
    
    # Model and processors
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
        device_map="auto" if device == "cuda" else {"": device},  # Force model to specific GPU, unless device is "cuda"
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Enable Flash Attention 2
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Create output file
    with open(export_json_path, "w") as f:
        pass
    
    # Process each question
    for question in questions:
        text_prompt = question["text"]
        images_list = get_data(image_folder_path, question["video"], data_type="rgb", sample_rate=sample_rate)
        depths_list = get_data(image_folder_path, question["video"], data_type="depth", sample_rate=sample_rate)
        poses_list = get_data(image_folder_path, question["video"], data_type="pose", sample_rate=sample_rate)

        
        print(f"========== Processing question: {question['question_id']} ==========")
        
        # Before processing each question
        gc.collect()
        torch.cuda.empty_cache()
        
        # Run inference
        output_text = kimivl_video_test(
            model,
            processor,
            image_paths=images_list,
            text_prompt=text_prompt,
            model_path=model_path,
            device=device
        )
        
        # Save output
        save_output(output_text, question, export_json_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--answer_file", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--export_json", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-VL-A3B-Thinking")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=5)
    args = parser.parse_args()
    
    main(
        question_file_path=args.question_file,
        answer_file_path=args.answer_file,
        image_folder_path=args.image_folder,
        export_json_path=args.export_json,
        model_path=args.model_path,
        device=args.device,
        sample_rate=args.sample_rate
    ) 