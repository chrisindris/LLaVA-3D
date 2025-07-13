# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
import os
import torch
import json
import gc
from typing import Literal, NamedTuple, Optional, TYPE_CHECKING
from dataclasses import asdict
from vllm.multimodal.utils import fetch_image

if TYPE_CHECKING:
    from vllm import EngineArgs
    from vllm.lora.request import LoRARequest
    from PIL import Image


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


def add_answers_to_questions(questions: list, answer_str: str):
    """
    Add answers to questions.
    Args:
        questions (list): List of questions.
        answer_str (str): Answer string to add.
    Returns:
        list: List of questions with answers.
    """

    # load answer from json file
    with open(answer_str, "r") as f:
        answers = json.load(f)
    for i, question in enumerate(questions):
        question["answer"] = answers[i]["text"]
        question["type"] = answers[i]["type"]
    return questions


def save_output(output_text: str, question: dict, output_file_path: str):
    # Parse the output text to extract the reasoning and answer
    # This is a placeholder implementation. You may need to adjust it based on the actual output format.
    try:
        print(output_text)
        output_json = json.loads(output_text)
        reasoning = output_json.get("reason", "")
        answer = output_json.get("answer", "")
        # append this new json enry to the output file
        with open(output_file_path, "a") as f:
            if question is not None:
                json.dump(
                    {
                        "reason": reasoning,
                        "text": answer,
                        "question_id": question["question_id"],
                        "scene_name": question["video"],
                        "prompt": question["text"],
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
                f.write("\n")
            else:
                json.dump(
                    {
                        "reason": reasoning,
                        "text": answer,
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
                f.write("\n")
    except json.JSONDecodeError:
        print(f"Failed to parse output: {output_text}")
        with open(output_file_path, "a") as f:
            json.dump(
                {
                    "reason": "Failed to parse",
                    "text": output_text,
                    "question_id": question["question_id"],
                    "scene_name": question["video"],
                    "prompt": question["text"],
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
            f.write("\n")


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
        # add all jpg files in the folder to the image_paths list
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


# You can set the maximum tokens for a video through the environment variable VIDEO_MAX_PIXELS
# based on the maximum tokens that the model can accept.
# export VIDEO_MAX_PIXELS = 32000 * 28 * 28 * 0.9
def hf_qwen_video_test(
    image_paths: list, text_prompt: str, model_path: str, device: str = "cuda:2"
):
    # You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
    messages = [
        # Image
        ## Local file path
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": "file:///path/to/your/image.jpg"},
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        ## Image URL
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        # ## Base64 encoded image
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": "data:image;base64,/9j/..."},
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        # ## PIL.Image.Image
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": pil_image},
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        # ## Model dynamically adjusts image size, specify dimensions if required.
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": "file:///path/to/your/image.jpg",
        #                 "resized_height": 280,
        #                 "resized_width": 420,
        #             },
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        # Video
        # Local video frames
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": image_paths,
                        # "resized_height": 280,
                        # "resized_width": 280,
                        "min_pixels": 4 * 28 * 28,
                        "max_pixels": 256 * 28 * 28,
                        "total_pixels": 20480 * 28 * 28,
                        "fps": 30.0,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a help assistant to answer the question concisely with your separate reasoning trace.",
                    }
                ],
            },
        ],
        ## Model dynamically adjusts video nframes, video height and width. specify args if required.
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": "file:///path/to/video1.mp4",
        #                 "fps": 2.0,
        #                 "resized_height": 280,
        #                 "resized_width": 280,
        #             },
        #             {"type": "text", "text": "Describe this video."},
        #         ],
        #     }
        # ],
    ]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    # print(model.hf_device_map)
    # print(model.device)
    processor = AutoProcessor.from_pretrained(model_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    # print("preprocess done")
    with torch.no_grad():
        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

        # Generate the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    gc.collect()
    torch.cuda.empty_cache()
    return output_text


def parse_json(json_output) -> str:
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(
                lines[i + 1 :]
            )  # Remove everything before "```json"
            json_output = json_output.split("```")[
                0
            ]  # Remove everything after the closing "```"
            # remove [ and ] if lines[i+1] is [
            if lines[i + 1] == "[":
                # remove the first [ and last ] from the json_output which occupies the first and last line
                json_output = json_output[1:-2]
                print("after removing: ", json_output)
            break  # Exit the loop once "```json" is found
    return json_output


def sglang_qwen_video_test(
    image_paths: list, text_prompt: str, model_path: str, port: int = 8080
):
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")

    class Answer(BaseModel):
        reason: str = Field(..., description="reason")
        answer: str = Field(..., description="answer")

    response = client.chat.completions.create(
        model=model_path,
        messages=[
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": path},
                        }
                        for path in image_paths
                    ],
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=128,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "qwen_output",
                "schema": Answer.model_json_schema(),
            },
        },
    )

    response_content = response.choices[0].message.content
    final_json_output = Answer.model_validate_json(response_content)
    return final_json_output.model_dump_json()


class ModelRequestData(NamedTuple):
    engine_args: "EngineArgs"
    prompt: str
    image_data: list["Image.Image"]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list["LoRARequest"]] = None


def load_qwen2_5_vl(
    model_name: str, question: str, image_paths: list[str], device_nums: int = 1
) -> ModelRequestData:
    try:
        from qwen_vl_utils import smart_resize
    except ModuleNotFoundError:
        print(
            "WARNING: `qwen-vl-utils` not installed, input images will not "
            "be automatically resized. You can enable this functionality by "
            "`pip install qwen-vl-utils`."
        )
        smart_resize = None

    # Set a fixed number of tokens per image.
    tokens_per_image = 128
    patch_size = 28 * 28
    max_pixels = tokens_per_image * patch_size

    # Calculate the required model length.
    num_images = len(image_paths)
    total_image_tokens = num_images * tokens_per_image

    # Add a buffer for the text prompt and generated output.
    text_and_generation_buffer = 1024
    calculated_max_model_len = total_image_tokens + text_and_generation_buffer

    # The context window of the model
    model_absolute_max_len = 32768

    if calculated_max_model_len > model_absolute_max_len:
        print(
            f"WARNING: The number of images ({num_images}) results in a required context "
            f"length ({calculated_max_model_len}) that exceeds the model's maximum "
            f"({model_absolute_max_len}). The model may not be able to process all images."
        )
        # reduce the number of images to the maximum number of images that can be processed by the model
        num_images = (
            model_absolute_max_len - text_and_generation_buffer
        ) // tokens_per_image
        final_max_model_len = model_absolute_max_len
    else:
        final_max_model_len = calculated_max_model_len

    engine_args = EngineArgs(
        model=model_name,
        tensor_parallel_size=device_nums,
        max_model_len=final_max_model_len,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": num_images},
    )

    # Update image paths to the number of images that can be processed by the model
    # get num_images images uniformly from whole image_paths
    if num_images < len(image_paths):
        # Uniformly sample num_images from the original list
        step = len(image_paths) / num_images
        indices = [int(i * step) for i in range(num_images)]
        used_image_paths = [image_paths[i] for i in indices]
    else:
        used_image_paths = image_paths

    placeholders = [{"type": "image", "image": path} for path in used_image_paths]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_name)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    def get_image(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    if smart_resize is None:
        print(
            "WARNING: `smart_resize` not available. Images will not be resized, which may lead to errors."
        )
        image_data = [get_image(path) for path in used_image_paths]
    else:

        def post_process_image(image: Image.Image) -> Image.Image:
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height, width, max_pixels=max_pixels
            )
            return image.resize((resized_width, resized_height))

        image_data = [post_process_image(get_image(path)) for path in used_image_paths]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


def vllm_qwen_video_test(
    image_paths: list, text_prompt: str, model_path: str, device_nums: int = 1
):
    """
    Run Qwen2.5-VL model using VLLM for multiple image paths to generate output.
    """

    class Answer(BaseModel):
        reason: str = Field(..., description="reason")
        answer: str = Field(..., description="answer")

    json_schema = Answer.model_json_schema()
    guided_decoding_params_json = GuidedDecodingParams(json=json_schema)
    req_data = load_qwen2_5_vl(model_path, text_prompt, image_paths, device_nums)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop_token_ids=req_data.stop_token_ids,
        guided_decoding=guided_decoding_params_json,
    )
    llm = LLM(**asdict(req_data.engine_args))

    outputs = llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {"image": req_data.image_data},
        },
        sampling_params=sampling_params,
        lora_request=req_data.lora_requests,
    )

    final_output = Answer.model_validate_json(outputs[0].outputs[0].text)

    return final_output.model_dump_json()


def main(
    question_file_path: str,
    answer_file_path: str,
    image_folder_path: str,
    export_json_path: str,
    model_path: str,
    inference_type: Literal["qwen", "sglang", "vllm"] = "qwen",
    port: int = 30000,
):
    # Load questions from a JSON file
    questions = load_questions_json(question_file_path)
    # Load answers from a JSON file
    questions = add_answers_to_questions(questions, answer_file_path)
    # Find image paths in questions
    sample_rate = 10
    questions = find_image_paths(questions, image_folder_path, sample_rate)

    for question in questions:
        # Get the image paths for the first question
        image_paths = question["scene_images_path"]
        # Get the text prompt for the first question
        question_text = question["text"]
        # Get text prompt from the question
        text_prompt = (
            question_text
            + " Please reason step by step, and give your reason and answer in the json format with field reason and answer."
        )
        # Run the Qwen video test
        if inference_type == "hf":
            output_text = hf_qwen_video_test(image_paths, text_prompt, model_path)
            output = parse_json(output_text[0])
        elif inference_type == "sglang":
            output_text = sglang_qwen_video_test(
                image_paths, text_prompt, model_path, port
            )
            output = output_text
        elif inference_type == "vllm":
            # from os environ CUDA_VISIBLE_DEVICES get available cuda nums. If not set, use 1
            device_nums = int(
                os.environ.get("CUDA_VISIBLE_DEVICES", "1").count(",") + 1
            )
            output_text = vllm_qwen_video_test(
                image_paths, text_prompt, model_path, device_nums=device_nums
            )
            output = output_text
        else:
            raise ValueError(
                "Invalid inference type. Choose from 'hf', 'sglang', or 'vllm'."
            )

        save_output(output, question, export_json_path)


if __name__ == "__main__":
    # Load images from a folder
    question_file_path = "/data/SceneUnderstanding/7792397/ScanQA_format/SQA_em1-below-35_formatted_LLaVa3d.json"
    answer_file_path = "/data/SceneUnderstanding/7792397/ScanQA_format/SQA_em1-below-35_formatted_LLaVa3d_answers.json"
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    image_folder_path = "/data/SceneUnderstanding/ScanNet/scans"
    export_path = "./qwen2.5vl_3d_test_results_vllm.json"
    port = 30000
    inference_type = "vllm"  # Choose from 'qwen', 'sglang', or 'vllm'
    if inference_type == "sglang":
        from openai import OpenAI
        from pydantic import BaseModel, Field
    elif inference_type == "hf":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    elif inference_type == "vllm":
        from vllm import LLM, SamplingParams, EngineArgs
        from vllm.sampling_params import GuidedDecodingParams
        from pydantic import BaseModel, Field
        from PIL import Image
        from transformers import AutoProcessor
    else:
        raise ValueError(
            "Invalid inference type. Choose from 'hf', 'sglang', or 'vllm'."
        )

    main(
        question_file_path,
        answer_file_path,
        image_folder_path,
        export_path,
        model_path,
        inference_type,
        port,
    )
