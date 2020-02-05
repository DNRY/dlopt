# 딥러닝을 위한 최적화와 수치해석
<img src="./images/book_cover.jpg" height="200">

## 목차
1. 개발환경 설정하기
1. [주피터 노트북과 파이썬 튜토리얼](https://github.com/DNRY/dlopt/blob/master/notebooks/CH02.ipynb)
1. [텐서플로 튜토리얼](https://github.com/DNRY/dlopt/blob/master/notebooks/CH03.ipynb)
1. [최적화 이론에 필요한 선형대수와 미분](https://github.com/DNRY/dlopt/blob/master/notebooks/CH04.ipynb)
1. [딥러닝에 필요한 최적화 이론](https://github.com/DNRY/dlopt/blob/master/notebooks/CH05.ipynb)
1. [고전 수치최적화 알고리즘](https://github.com/DNRY/dlopt/blob/master/notebooks/CH06.ipynb)
1. [딥러닝을 위한 수치최적화 알고리즘](https://github.com/DNRY/dlopt/blob/master/notebooks/CH07.ipynb)
1. [선형회귀 모델](https://github.com/DNRY/dlopt/blob/master/notebooks/CH08.ipynb)
1. [선형 분류 모델](https://github.com/DNRY/dlopt/blob/master/notebooks/CH09.ipynb)
1. [신경망 회귀 모델](https://github.com/DNRY/dlopt/blob/master/notebooks/CH10.ipynb)
1. [신경망 분류 모델](https://github.com/DNRY/dlopt/blob/master/notebooks/CH11.ipynb)
1. [언더피팅/오버피팅 소개](https://github.com/DNRY/dlopt/blob/master/notebooks/CH12.ipynb)
1. [언더피팅의 진단과 해결책](https://github.com/DNRY/dlopt/blob/master/notebooks/CH13.ipynb)
1. [오버피팅의 진단과 해결책](https://github.com/DNRY/dlopt/blob/master/notebooks/CH14.ipynb)
1. [텐서보드(TensorBoard) 활용](https://github.com/DNRY/dlopt/blob/master/notebooks/CH15.ipynb)
1. [모델 저장하기와 불러오기](https://github.com/DNRY/dlopt/blob/master/notebooks/CH16.ipynb)
1. 딥러닝 가이드라인
1. [CNN 모델](https://github.com/DNRY/dlopt/blob/master/notebooks/CH18.ipynb)
1. [GAN(Generative Adversarial Networks) 모델](https://github.com/DNRY/dlopt/blob/master/notebooks/CH19.ipynb)
1. [영상](https://github.com/DNRY/dlopt/blob/master/notebooks/CH20.ipynb)
1. [문자열 분석 word2vec](https://github.com/DNRY/dlopt/blob/master/notebooks/CH21.ipynb)

## Acaconda Environment
### Without `yml`
```bash
$ conda remove --name deep-learning --all
$ conda create --name deep-learning python=3.5
$ conda activate deep-learning
(deep-learning) $ conda install numpy=1.14.5 tensorflow=1.10 matplotlib=2 jupyter_client=5.3.1 jupyter notebook seaborn scikit-learn setuptools=39.1.0 cython
```

### With `yml`
```
$ conda remove --name deep-learning --all $ conda env create -f env.yml
$ conda activate deep-learning
```
