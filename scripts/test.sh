num_layers=7

CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=62202 test.py \
        --datapath ../dataset \
        --logpath test_1_test \
        --benchmark pascal \
        --backbone resnet50 \
        --fold 3 \
        --condition mask \
        --num_query 50 \
        --num_layers $num_layers \
        --epochs 50 \
        --lr 1e-4 \
        --bsz 1 \
        --nshot 1