# The vanilla expermient from the LLaVA-3D repo

export CUDA_VISIBLE_DEVICES=0

# --- set the embodiedscan file ---
unlink playground/data/annotations/embodiedscan_infos.json
ln -s /data/SceneUnderstanding/SU_cursor/LLaVA-3D/playground/data/annotations/embodiedscan_infos_full_formatted_cluster.json /data/SceneUnderstanding/SU_cursor/LLaVA-3D/playground/data/annotations/embodiedscan_infos.json

python llava/eval/run_llava_3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --video-path /data/SceneUnderstanding/ScanNet/scans/scene0000_00/ \
    --query "Describe this scene." 
