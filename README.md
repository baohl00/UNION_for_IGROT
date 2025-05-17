# UNION: A Lightweight Target Representation for Efficient Image-Guided Retrieval with Optional Textual Queries 

## Abstract

Image-Guided Retrieval with Optional Text (IGROT) is a general retrieval setting where a query consists of an anchor image, with or without accompanying text, aiming to retrieve semantically relevant target images. This formulation unifies two major tasks: Composed Image Retrieval (CIR) and Sketch-Based Image Retrieval (SBIR). In this work, we address IGROT under low-data supervision by introducing UNION, a lightweight and generalizable target representation that fuses the image embedding with a null-text prompt. Unlike traditional approaches that rely on frozen target features, UNION enhances semantic alignment with multimodal queries while requiring no architectural modifications to pretrained vision-language models. With only 5k training samples—from LlavaSCo for CIR and Training-Sketchy for SBIR—our method achieves competitive results across benchmarks, including CIRCO R@50 of 34.4 and Sketchy mAP@200 of 83.0, surpassing many heavily supervised baselines. This demonstrates the robustness and efficiency of UNION in bridging vision and language across diverse query types.

## Setup
```
micromamba create -n cir python=3.9
micromamba activate cir
pip install -r requirements_blip.txt 
pip install -r requirements.txt
```

### Model Download & Data Preparation
Please follow the instruction in [here](https://github.com/google-deepmind/magiclens/blob/main/data/README.md).

## Inference
We prepare two different files for inference stage. You can choose base or large version of UNION, if you run this:
```
bash scripts/inference.sh
```
or run both two versions:   
```
bash scripts/fast.sh
```

Due to the weight conversion, the performance may be slightly different:

In **Zero-Shot Composed Image Retrieval**: `FashionIQ`, `CIRR`and `CIRCO`

| Model | map@5 | map@10 | map@25 | map@50 |
|----------|----------|----------|----------|----------|
| Prior SOTA | 26.8 | 27.6 | 30.0 | 31.0 |
| Base (original) | 23.1 | 23.8 | 25.8 | 26.7 |
| Base (reproduced) | 25.5 | 26.5 | 28.5 | 29.4 |
| Base + _VisionProjector_ | 25.7 | 26.7 | 28.6 | 29.6 |
| Large (original) | 29.6 | 30.8 | 33.4 | 34.4 |
| Large (reproduced) | _30.1_ | _31.4_ | _33.8_ | _34.9_ |
| Large + _VisionProjector_ (SOTA) | **35.8** | **36.8** | **39.3** | **40.4** |

In **Zero-Shot Sketch-Based Image Retrieval**: `Sketchy`, `TUBerlin` and `QuickDraw`

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

```

## Acknowledgement 

We extend our gratitude to the open-source efforts of [TransAgg](https://github.com/Code-kunkun/ZS-CIR). 
