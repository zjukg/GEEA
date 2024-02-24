## GEEA

This repository is the official implementation of [Revisit and Outstrip Entity Alignment: A Perspective of Generative Models](https://arxiv.org/abs/2305.14651), ICLR 2024.


### Quick Start

**1. Download Datasets**

Download datasets from [MCLEA](https://github.com/lzxlin/MCLEA) and [MEAformer](https://github.com/zjukg/MEAformer).


**2. Install Packages**

```
pip install -r requirement.txt
```

**3. Run Script**

```
sh run_geea.sh 0 DBP15K zh-en 0.3 
```

**4. For Other Datasets**

```
sh run_geea.sh 0 DBP15K ja-en 0.3 
sh run_geea.sh 0 DBP15K fr-en 0.3 
sh run_geea.sh 0 FBDB15K norm 0.2
sh run_geea.sh 0 FBYG15K norm 0.2
```

## Special Thanks

This source code is adaped from the official repository of [MEAformer](https://github.com/zjukg/MEAformer).

## Citation

```bib
@inproceedings{GEEA,
    author = {Lingbing Guo and
              Zhuo Chen and
              Jiaoyan Chen and
              Yin Fang and
              Wen Zhang and
              Huajun Chen},
    title = {Revisit and Outstrip Entity Alignment: {A} Perspective of Generative Models},
    booktitle = {ICLR},
    year = {2024}
}
```
