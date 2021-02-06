The IEGN model for Dynamic Recommendation
The implementation of the paper:

 "**IEGN: Interest Evolution-driven Gated Neighborhood Aggregation for Dynamic Recommendation**", in the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (**KDD 2021**)* 

**Please cite our paper if you use our code. Thanks!**
## Environments

- python 3.6
- PyTorch (version: 0.4.0)
- numpy (version: 1.19.5)
- scipy (version: 1.4.0)


## Dataset

The *Amazon-Beauty* and *Amazon-CDs* datasets are from http://jmcauley.ucsd.edu/data/amazon/, and *UserBehavior* datasets is from https://tianchi.aliyun.com/dataset/dataDetail?dataId=649. 


## Example to run the code

Train and evaluate the model (you are strongly recommended to run the program on a machine with GPU):

```
python run.py
```

## Acknowledgment

The implemention of graph is modified based on *[this](https://github.com/kuandeng/LightGCN)*
The implemention of dynamic recommendation is is heavily built on *[this](https://github.com/allenjack/HGN)*

Thanks for their amazing works.



















