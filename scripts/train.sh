CUDA_VISIBLE_DEVICES=0 python train_LT.py -e cnnt/DeepCAD -m cnnt_vq
CUDA_VISIBLE_DEVICES=0 python train_LT.py -e vhp/DeepCAD -m vhp_vq

CUDA_VISIBLE_DEVICES=0 python encode_LT.py -e vhp/DeepCAD -m vhp_vq
CUDA_VISIBLE_DEVICES=0 python encode_LT.py -e cnnt/DeepCAD -m cnnt_vq

CUDA_VISIBLE_DEVICES=0 python train_LT.py -e gpt/DeepCAD -m gpt



