[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Robust Unstructured Knowledge Access in Conversational Dialogue with ASR Errors

This repository contains the data, scripts and source codes for [DSTC10](https://dstc10.dstc.community/home/) Track 2, task 2 submission from New York University Shanghai.

Due to time for intensive code cleaning, we only share the training and evaluation source code for knowledge cluster classification part with the proposed error simulator, which is the heart of our paper accepted in the AAAI 2022 workshop. I will release other components such as knowledge title clustering and NER to this repository in the future.
Our code base is derived from https://github.com/alexa/alexa-with-dstc10-track2-dataset, and introduced a new task. If you use this repository, please cite the following article:
```
@inproceedings{tam2022sim,
  title={Robust Unstructured Knowledge Access in Conversational Dialogue with ASR Errors},
  author={Yik-Cheung Tam and Jiacheng Xu and Jiakai Zou and Zecheng Wang and Tinglong Liao and Shuhan Yuan},
  booktitle = "DSTC10 Workshop @ AAAI",
  month = Feb,
  year={2022}
}
```
