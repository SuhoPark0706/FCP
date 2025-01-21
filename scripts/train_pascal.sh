GPUNUM=1
PORTNUM=$((1090+GPUNUM))

PROMPT_LOSS=1.0
mask_loss=0.5
attn_loss=0.05
attn_drop_out=0.5
num_layers=3
spt_num_query=50

for FOLDNUM in 0 1 2 3
do
CUDA_VISIBLE_DEVICES=$GPUNUM python3 -m torch.distributed.launch --nproc_per_node=1  --master_port=$PORTNUM train.py \
        --datapath ../dataset \
        --logpath pascal-113-spt_pro_num_${spt_num_query}-fold$FOLDNUM \
        --benchmark pascal \
        --backbone resnet50 \
        --fold $FOLDNUM \
        --seed 321 \
        --num_layers $num_layers \
        --prompt_loss $PROMPT_LOSS \
        --mask_loss $mask_loss \
        --attn_loss $attn_loss \
        --attn_drop_out $attn_drop_out \
        --condition mask \
        --num_query 50 \
        --spt_num_query $spt_num_query \
        --nworker 2 \
        --epochs 100 \
        --lr 2e-4 \
        --bsz 8 \
        # --use_log 
done