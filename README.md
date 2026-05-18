# MMRest
[CVPR2026]An official implementation of "Multi-Metric Representation Learning Strategy Based on Clustering for Fine-Grained Multimodal Sentiment Analysis" in PyTorch.


### Usage
1.Download the CMU-MOSI and CMU-MOSEI dataset from [Google Drive](https://drive.google.com/drive/folders/1djN_EkrwoRLUt7Vq_QfNZgCl_24wBiIK) or [Baidu Disk](https://pan.baidu.com/share/init?surl=Wxo4Bim9JhNmg8265p3ttQ) (extraction code: g3m2)  
Place them under the folder ```MMRest/datasets```

2.Set up the environment (need conda prerequisite)
```
conda env create -f environment.yml
conda activate MMRest
```
3. starting 
```python
python main.py
```

### Acknowledgements

This work builds upon two outstanding open-source projects: [MMIM](https://github.com/declare-lab/Multimodal-Infomax) and [MCL-MCF](https://github.com/Zhudogsi/MCL-MCF).

We are deeply grateful to the authors for making their code publicly available. 

Please also cite the corresponding papers if you find our work useful:

```bibtex
@inproceedings{han2021improving,
  title={Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis},
  author={Han, Wei and Chen, Hui and Poria, Soujanya},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={9180--9192},
  year={2021}
}

@article{DBLP:journals/taffco/FanZTYXL25,
  author       = {Cunhang Fan and
                  Kang Zhu and
                  Jianhua Tao and
                  Guofeng Yi and
                  Jun Xue and
                  Zhao Lv},
  title        = {Multi-Level Contrastive Learning: Hierarchical Alleviation of Heterogeneity
                  in Multimodal Sentiment Analysis},
  journal      = {{IEEE} Trans. Affect. Comput.},
  volume       = {16},
  number       = {1},
  pages        = {207--222},
  year         = {2025}
}
