# Use CUDA_VISIBLE_DEVICES=0,1, ... to specify the devices
python -m torch.distributed.launch \
    --nproc_per_node=2 train.py