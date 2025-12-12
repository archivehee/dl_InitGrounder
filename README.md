[2025.11] Principles of Deep Learning - Short Project

# "InitGrounder" 

Online User Cold-Start Recommendation with Popularity-Aware Learning

## Our Dataset Split
| Domain | User | Items | Train | Valid | Test |
| :--- | ---: | :---: | ---: | :---: | ---: |
| Electronics | 3,000 | 35,002 | 49,525 | 3,000 | 3,000 |
| Home and Kitchen | 3,000 | 41,138 | 51,141 | 3,000 | 3,000 |

## Raw Dataset is downloaded by https://amazon-reviews-2023.github.io/
| Domain | User | Items | Reviews |
| :--- | ---: | :---: | ---: |
| Electronics | 18.3M | 1.6M | 2.7B |
| Home and Kitchen | 23.2M | 3.7M | 3.1B |

### We filtered out for 3,000 users by following process (RAW -> Our dataset split)
1. overlapped user 3,000 selection - data/amazon/overlap_usr_selec.py (output: f_usr_id.json)
2. filtered interaction & item meta data generation -  data/amazon/rev_itm_sample.py
3. Datasplit ->
   1) inter_cdr(CDR)
   - Ours Model evaluation (same user index, respective item index for each domain) 
   3) lgn_cdr(train/valid/test split to use LightGCN as CDR method)
   - Same user index, unified item index 0~35,001 (Electronics) / 35,002~76,139 (Home and Kitchen)
  

## Backbone Model
LightGCN (SIGIR, 2020) from
https://github.com/gusye1234/LightGCN-PyTorch
