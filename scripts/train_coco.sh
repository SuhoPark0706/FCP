FOLDNUM=3
GPUNUM=3

PORTNUM=$((1010+GPUNUM))

PROMPT_LOSS=1.0
mask_loss=0.5
attn_loss=0.05
attn_drop_out=0.5
num_layers=3

CUDA_VISIBLE_DEVICES=$GPUNUM python3 -m torch.distributed.launch --nproc_per_node=1  --master_port=$PORTNUM train.py \
        --datapath ../dataset \
        --logpath vgg-coco-numLayers_$num_layers-attnLoss_$attn_loss-maskLoss_$mask_loss-dropout_$attn_drop_out-fold$FOLDNUM \
        --benchmark coco \
        --backbone vgg16 \
        --fold $FOLDNUM \
        --seed 321 \
        --num_layers $num_layers \
        --prompt_loss $PROMPT_LOSS \
        --mask_loss $mask_loss \
        --attn_loss $attn_loss \
        --attn_drop_out $attn_drop_out \
        --condition mask \
        --num_query 50 \
        --nworker 4 \
        --epochs 50 \
        --lr 1e-4 \
        --bsz 8 \
        --use_log 