# ResFlow

This repository contains the Tensorflow implementation for "ResFlow" as well as other baseline methods for KDD 2024 paper entitled "Residual Multi-Task Learner for Applied Ranking".

## Datasets
We use three public datasets: Ali-CCP, AE, KuaiPure and MovieLens.
* Ali-CCP is a benchmark dataset for conversion rate prediction, collected from traffic logs in Taobao platform, please refer to [official page](https://tianchi.aliyun.com/dataset/408) for more details.
* AE is a dataset gathered from real-world traffic logs of the search system in AliExpress.please refer to [official page](https://tianchi.aliyun.com/dataset/74690) for more details.
* KuaiRand-Pure is an unbiased sequential recommendation dataset collected from the recommendation logs of the video-sharing mobile app, Kuaishou. Please refer to [official page](https://github.com/chongminggao/KuaiRand) for more details.
* MovieLens is a rating dataset from the MovieLens web site. Please refer to [official page](https://grouplens.org/datasets/movielens/) for more details. 
<div> 
All datasets above were processed on our internal platform and may not suitable for general personal system or environment.
</div>

## Run the code
The code is only tested on our internal platform and may not be able to executed on general system without minor modification and adaption, This code is provided for reference purpose temporarily. We will work on an easy-to-run version in the future.
* To reproduce the experiments on e-commercial datasets, please refer to e-comm directory
* To reproduce the experiments on other modalities, please refer to other-modality directory
