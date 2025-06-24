# Our own questions, to ensure that the initial is not merely a good example.

export CUDA_VISIBLE_DEVICES=0

python /root/SceneUnderstanding/LLaVA-3D/llava/eval/run_llava_3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --image-file https://llava-vl.github.io/static/images/view.jpg \
    --query "Describe this image"

# by default, at most 24 frames are used by the video processor.
