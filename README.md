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


## Backbone Model
LightGCN (SIGIR, 2020) from
https://github.com/gusye1234/LightGCN-PyTorch
