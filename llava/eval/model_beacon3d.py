import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_videos, get_model_name_from_path

from PIL import Image
import math

import pdb


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    
    # Model
    #breakpoint()
    
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Override decoding parameters if paper defaults are requested
    if args.use_paper_decoding_params:
        print("INFO: Using paper decoding parameters: Temperature=0.0, Top_p=1.0")
        args.temperature = 0.0
        args.top_p = 1.0 # Explicitly set, though None might also work as 1.0 if sampling
    
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    # ans_file = open(answers_file, "w")
    ans_list = [] # a list of dictionaries, to be written after to the file

    for line in tqdm(questions): # The key info for the questions file
        #breakpoint()
        idx = line["question_id"]
        video_file = line["scene_id"]
        video_path = os.path.join(args.video_folder, video_file)
        qs = line["question"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        
        # Override system prompt if provided
        if args.override_system_prompt is not None:
            print(f"INFO: Overriding system prompt with: '{args.override_system_prompt}'")
            conv.system = args.override_system_prompt
            
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        tokenizer_call_kwargs = {}
        if args.no_prompt_tokenizer_truncation:
            tokenizer_call_kwargs['truncation'] = False
            # tokenizer_call_kwargs['max_length'] = None # Not strictly needed if truncation is False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt', **tokenizer_call_kwargs).unsqueeze(0).cuda()

        videos_dict = process_videos( # just passing in the scene on the fly; tensors containing the images, depths, poses (4x4), and intrinsics (4x4)
            video_path,
            processor['video'],
            mode=args.frame_selection_mode,
            device=model.device,
            text=cur_prompt
        )

        images_tensor = videos_dict['images'].to(model.device, dtype=torch.bfloat16)
        depths_tensor = videos_dict['depths'].to(model.device, dtype=torch.bfloat16)
        poses_tensor = videos_dict['poses'].to(model.device, dtype=torch.bfloat16)
        intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                depths=depths_tensor,
                poses=poses_tensor,
                intrinsics=intrinsics_tensor,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                eos_token_id=tokenizer.eos_token_id
            )


        outputs = tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        ans_id = shortuuid.uuid()
        ans_list.append({"question_id": idx,
                         "scene_id": video_file,
                         "prompt": cur_prompt,
                         "text": outputs,
                         "answer_id": ans_id,
                         "model_id": model_name,
                         "metadata": {}})

        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file = open(answers_file, "w")
    json.dump(ans_list, ans_file, indent=4)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava3d-v1.5-7b-task-v3-tuning")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="playground/data/LLaVA-3D-Pretrain")
    parser.add_argument("--question-file", type=str, default="playground/data/annotations/llava3d_sqa3d_val_question.json")
    parser.add_argument("--answers-file", type=str, default="./llava3d_sqa3d_val_answer_pred.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1") # The conversation mode
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=0.5)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--frame_selection_mode", type=str, default="random", choices=["random", "uniform"], help="Mode for selecting video frames. 'random' or 'uniform'.")
    parser.add_argument("--no_prompt_tokenizer_truncation", action='store_true', help="If set, disables truncation in the tokenizer for the main prompt.")
    # New arguments for decoding and system prompt
    parser.add_argument("--use_paper_decoding_params", action='store_true', help="If set, uses Temperature=0.0 and Top_p=1.0.")
    parser.add_argument("--override_system_prompt", type=str, default=None, help="Override the default system prompt of the conversation mode.")
    args = parser.parse_args()
    
    eval_model(args)