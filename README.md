# UNION: A Lightweight Target Representation for \\Efficient Image-Guided Retrieval with Optional Textual Queries 

## Abstract

Composed Image Retrieval (CIR) is a specific task related to visual information retrieval using a query which is comprised of both image and text data. Following this idea, Zero-Shot CIR aims to tackle CIR task without pre-training the model on the training labeled triplets consisting of the query image, the textual description, and the target image. The architecture of state-of-the-art (SOTA) methods compose of two parts: the encoding stage extracting key features and the core model based on Transformer or Contrastive Learning in order to match the key features. This popular common gives us a question: _Is there any way to leverage the model directly instead of improving data quantity or building a more complex architecture?_ Therefore, in this paper, we demonstrate **Vision Projector** (VP) - an efficient and simple, easy-to-plug projector that improve remarkably the released model - MagicLens. Without pre-training them one more time, VP is directly attached into MagicLens and increase baseline performance throughout all experimental datasets, and particularly, in the CIRCO dataset, increase about 18\% the metric scores despite of not using their best model and achieve the SOTA scores.

<blockquote>
We introduce MagicLens, a series of self-supervised image retrieval models that support
open-ended instructions. The core thesis of MagicLens is that text
instructions can enable retrieving images with
richer relations beyond visual similarity. MagicLens is built on a
key novel insight: image pairs that naturally occur
on the same web pages contain a wide range of implicit relations (e.g., inside view of), and we
can bring those implicit relations explicit by synthesizing instructions via large multimodal models (LMMs) and large language models (LLMs).
Trained on 36.7M (query image, instruction, target image) triplets with rich semantic relations
mined from the web, MagicLens achieves comparable or better results on eight benchmarks of
various image retrieval tasks than prior state-of-the-art (SOTA) methods. Remarkably, it outperforms previous SOTA but with a 50× smaller
model size on multiple benchmarks. Additional
human analyses on a 1.4M-image unseen corpus
further demonstrate the diversity of search intents
supported by MagicLens.
![Intro image](https://open-vision-language.github.io/MagicLens/static/images/magiclens_overview.png)
</blockquote>

## Setup
```
conda create --name magic_lens python=3.9
conda activate magic_lens
git clone https://github.com/google-research/scenic.git
cd scenic
pip install .
pip install -r scenic/projects/baselines/clip/requirements.txt
# you may need to install corresponding GPU version of jax following https://jax.readthedocs.io/en/latest/installation.html
# e.g.,
# # CUDA 12 installation
# Note: wheels only available on linux.
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # CUDA 11 installation
# Note: wheels only available on linux.
# pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Model Download & Data Preparation
Please follow the instruction in [here](https://github.com/google-deepmind/magiclens/blob/main/data/README.md).

## Inference
We prepare two different files for inference stage. You can choose base or large version of MagicLens, if you run this:
```
bash scripts/inference.sh
```
or run both two versions:   
```
bash scripts/fast.sh
```

Due to the weight conversion, the performance may be slightly different:

In `FashionIQ`

| Model | R@10 | R@50 | 
|----------|----------|----------|
| SOTA | **46.2** | **67.3** |
| Base (original) | 26.3 | 47.4 |
| Base (reproduced) | 26.4 | 48.6 |
| Base + _VisionProjector_ | 26.6 | 48.6 |
| Large (original) | 30.7 | 52.5 |
| Large (reproduced) | 30.8 | 52.1 | 
| Large + _VisionProjector_ | _32.1_ | _53.1_ |

In `CIRR`

| Model | R@1 | R@5 | R@50 |
|----------|----------|----------|----------|
| SOTA |  **37.9** | **68.9** | **93.9** |
| Base (original) | 27.0 | 58.0 | 91.1 |
| Base (reproduced) | 31.3 | 61.5 | 92.1 |
| Base + _VisionProjector_ | 31.5 | 61.8 | 92.0 |
| Large (original) | 30.1 | 61.7 | 92.6 |
| Large (reproduced) | 33.3 | 63.8 | _93.1_ |
| Large + _VisionProjector_ | _35.7_ | _65.1_ | 92.8 |  

_(*): The inference code for CIRR dataset is not published._

In `CIRCO`
| Model | map@5 | map@10 | map@25 | map@50 |
|----------|----------|----------|----------|----------|
| Prior SOTA | 26.8 | 27.6 | 30.0 | 31.0 |
| Base (original) | 23.1 | 23.8 | 25.8 | 26.7 |
| Base (reproduced) | 25.5 | 26.5 | 28.5 | 29.4 |
| Base + _VisionProjector_ | 25.7 | 26.7 | 28.6 | 29.6 |
| Large (original) | 29.6 | 30.8 | 33.4 | 34.4 |
| Large (reproduced) | _30.1_ | _31.4_ | _33.8_ | _34.9_ |
| Large + _VisionProjector_ (SOTA) | **35.8** | **36.8** | **39.3** | **40.4** |

## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```latex
@inproceedings{zhang2024magiclens,
  title={MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions},
  author={Zhang, Kai and Luan, Yi and Hu, Hexiang and Lee, Kenton and Qiao, Siyuan and Chen, Wenhu and Su, Yu and Chang, Ming-Wei},
  booktitle={The Forty-first International Conference on Machine Learning (ICML)},
  year={2024},
  pages={to appear}
}
```

## Acknowledgement 

We extend our gratitude to the open-source efforts of [TransAgg](https://github.com/Code-kunkun/ZS-CIR). 
