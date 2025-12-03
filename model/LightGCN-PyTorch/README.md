cd ./model/LightGCN-PyTorch/code
# LGN single domain recommendation

# Electronics

## 999 neg sample
python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0 \
  --bpr_batch 256 \
  --recdim 64 \
  --layer 3 \
  --lr 0.001

[TEST] {'hr': array([0.03      , 0.04633333]), 'ndcg': array([0.01591893, 0.0200025 ])}


---------------------------------

# Home and Kitchen

## 999 neg sample

python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0 \
  --bpr_batch 256 \
  --recdim 64 \
  --layer 3 \
  --lr 0.001

[TEST] {'hr': array([0.01466667, 0.02433333]), 'ndcg': array([0.00772152, 0.01016337])}

  --save_pretrain 1


-------------------------------------
# LGN to CDR
## w/ mask 600 users
python main_mask.py --dataset lgn_cdr --model lgn --topks "[10,20]" --dropout 0 --bpr_batch 256 --recdim 64 --layer 3 --lr 0.001

## w/o mask 600 users
python main_mask.py --dataset lgn_cdr --model lgn --topks "[10,20]" --dropout 0 --bpr_batch 256 --recdim 64 --layer 3 --lr 0.001 --mask_users 0

