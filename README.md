# MMBS (Towards Robust Visual Question Answering: Making the Most of Biased Samples via Contrastive Learning)

Here is the implementation of our Findings of EMNLP-2022 [Towards Robust Visual Question Answering: Making the Most of Biased Samples via Contrastive Learning](http://arxiv.org/abs/2210.04563). 

This repository contains code modified from [here for SAR+MMBS](https://github.com/PhoebusSi/SAR/) and [here for SAR+LMH](https://github.com/chrisc36/bottom-up-attention-vqa), many thanks!

![image](https://github.com/PhoebusSi/MMBS/blob/main/qualitativeComparison.jpg)
Qualitative comparison of our method LMH+MMBS against the plain method UpDn and the debiasing method LMH. In VQA-CP v2 (upper), the question types (‘Does the’ and ‘How many’) bias UpDn to the most common answers (see Fig. 5 for the an- swer distribution). LMH alleviates the language priors for yesno questions (upper left), while it fails on the more difficult non-yesno questions (upper right). Be- sides, LMH damages the ID performance, giving an un- common answer to the common sample from VQA v2 (lower right). MMBS improves the OOD performance while maintains the ID performance (lower right).


![image](https://github.com/PhoebusSi/MMBS/blob/main/MMBS-model.jpg)

Overview of our method. The question cate- gory words are highlighted in yellow. The orange circle and blue triangle denote the cross-modality representa- tions of the original sample and positive sample. The other samples in the same batch are the negative sam- ples, which are denoted by the gray circles.

## Requirements
* python 3.7.6
* pytorch 1.5.0
* zarr
* tdqm
* spacy
* h5py

## More code will be released soon.

## Reference
If you found this code is useful, please cite the following paper:
```
@article{Si2022TowardsRV,
  title={Towards Robust Visual Question Answering: Making the Most of Biased Samples via Contrastive Learning},
  author={Qingyi Si and Yuanxin Liu and Fandong Meng and Zheng Lin and Peng Fu and Yanan Cao and Weiping Wang and Jie Zhou},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.04563}
}
```
