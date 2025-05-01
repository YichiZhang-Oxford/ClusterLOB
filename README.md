# ClusterLOB

We introduce **ClusterLOB**, a methodology designed to enhance trading performance by leveraging clustering techniques. Specifically, we classify market participants into three distinct clusters representing directional, opportunistic, and market-making traders, which allows for a more nuanced approach to market microstructure modelling and signal generation.

**Paper: [ClusterLOB: Enhancing Trading Strategies by Clustering Orders in Limit Order Books
](https://arxiv.org/abs/2504.20349).**

## Data

We gather our data through [LOBSTER](https://lobsterdata.com/) by Huang et al. (2011), which uses ITCH data from NASDAQ to reproduce the LOB for any stock on NASDAQ to any specified level.

Table: Summary of 15 stocks categorized by tick size along with their sector and market capitalization.

## Citation
If you find this repository helpful in your work, please cite our paper.
```bibTex
@misc{zhang2025clusterlobenhancingtradingstrategies,
      title={ClusterLOB: Enhancing Trading Strategies by Clustering Orders in Limit Order Books}, 
      author={Yichi Zhang and Mihai Cucuringu and Alexander Y. Shestopaloff and Stefan Zohren},
      year={2025},
      eprint={2504.20349},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2504.20349}, 
}
```
