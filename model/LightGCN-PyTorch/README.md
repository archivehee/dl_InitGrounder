cd /home/heek/aigs/InitGrounder/model/LightGCN-PyTorch/code
# Electronics

python main.py \
  --dataset "Electronics" \
  --model "lgn" \
  --topks "[10,20]"

## comb
### 999 neg sample
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

### 99 neg sample
python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0 \
  --bpr_batch 256 \
  --recdim 64 \
  --layer 3 \
  --lr 0.001



---------------------------------

# Home and Kitchen

python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]"


## comb
### 999 neg sample

python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0.1 \
  --bpr_batch 1024 \
  --recdim 128 \
  --layer 3 \
  --lr 0.0001

  [TEST] {'hr': array([0.00933333, 0.016     ]), 'ndcg': array([0.00464726, 0.00631437])}

python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0.1 \
  --bpr_batch 1024 \
  --recdim 128 \
  --layer 2 \
  --lr 0.001

  [TEST] {'hr': array([0.008, 0.015]), 'ndcg': array([0.00420698, 0.00593444])}

python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0 \
  --bpr_batch 1024 \
  --recdim 128 \
  --layer 3 \
  --lr 0.001
[TEST] {'hr': array([0.012, 0.02 ]), 'ndcg': array([0.00565367, 0.00767974])}

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

### 99 neg sample

python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0 \
  --bpr_batch 256 \
  --recdim 64 \
  --layer 3 \
  --lr 0.001 \

  [TEST] {'hr': array([0.06033333, 0.10166667]), 'ndcg': array([0.03379166, 0.04419272])}

  python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0 \
  --bpr_batch 256 \
  --recdim 64 \
  --layer 3 \
  --lr 0.001 \
  --save_pretrain 1