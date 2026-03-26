# 项目名称

拼音输入法


## 简介

实现了一个拼音转汉字引擎，包含横向的三个语料库对比和纵向的模型二元三元搜索对比

## 结构

2024010694/
├── data/                   
│   ├── 一二级汉字表.txt      
│   ├── 拼音汉字表.txt        
│   └── input.txt            
├── corpus/                  # 语料库 (未上传全量数据)
├── src/                     # 核心代码
│   ├── model.py             # 二元Viterbi 算法实现
│   └── pretraining.py       # 基础sina语料预训练

├── smp_train/
│   └── smp_main.py          # 加载smp、wiki语料库
├── 
└── README.md                # 本文件

## 使用方法





