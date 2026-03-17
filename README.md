# AViSal360: Audiovisual Saliency Prediction for 360º Video

Code and models for *“AViSal360: Audiovisual Saliency Prediction for 360º Video”* ([PDF](https://graphics.unizar.es/papers/ISMAR2024_AViSal360.pdf))


Edurne Bernal-Berdun, Jorge Pina, Mateo Vallejo, Ana Serrano, Daniel Martin, and Belen Masia

**ISMAR 2024**

## Abstract
Saliency prediction in 360º video plays an important role in modeling visual attention, and can be leveraged for content creation, compression techniques, or quality assessment methods, among others. Visual attention in immersive environments depends not only on visual input, but also on inputs from other sensory modalities, primarily audio. Despite this, only a minority of saliency prediction models have incorporated auditory inputs, and much remains to be explored about what auditory information is relevant and how to integrate it in the prediction. In this work, we propose an audiovisual saliency model for 360º video content, AViSal360. Our model integrates both spatialized and semantic audio information, together with visual inputs. We perform exhaustive comparisons to demonstrate both the actual relevance of auditory information in saliency prediction, and the superior performance of our model when compared to previous approaches.

Visit our [website](https://graphics.unizar.es/projects/AViSal360_2024/) for more information and supplementary material.

## AViSal360 Model
Our model can be downloaded at: [https://nas-graphics.unizar.es/s/avisal360_model](https://nas-graphics.unizar.es/s/avisal360_model)
AViSal360 was trained on the [D-SAV360](https://graphics.unizar.es/projects/D-SAV360/dataset_index.html) dataset. To ensure robust evaluation, we used k-fold cross-validation (k=5) during training. We provide all five models trained on different dataset splits, along with the corresponding video splits, which are located in the `k_folds` folder.

### Requirements

The code has been tested with:
```
librosa==0.10.1
matplotlib==3.7.5
numpy==1.21.6
pandas==1.4.3
opencv_python==4.5.4.58 
torch==1.13.1+cu116
torchvision==0.14.1+cu116
tqdm==4.64.0
```

### Proposed Audio Energy Maps
Details about our proposed audio energy maps are provided in the [supplementary material](https://nas-graphics.unizar.es/s/8WcJBwaJM2PNPE2). You can access the code for generating the AEMs at our GitHub repository: [AmbisonicPowermapTest](https://github.com/R3Ngfx/AmbisonicPowermapTest).

#### Training Data
Preprocessed training data can be found in this [folder](https://nas-graphics.unizar.es/s/data_AViSal360). If you use this data, please cite the dataset publication [D-SAV360](https://graphics.unizar.es/projects/D-SAV360/dataset_index.html). The folder includes the following data: audio segments used to extract LogMel spectrograms, the LogMel spectrograms provided as input to ImageBind, the audio embeddings produced by ImageBind, our proposed AEMs, extracted video frames, and the computed saliency maps.

---

### AViSal360 Inference

This script runs inference with a pretrained **AViSal360** model to generate audiovisual saliency maps for 360° videos.

#### **Important:**

You must use the **matching test set and model**:

* **Test videos 0 → Train model kfold 0**
* **Test videos 1 → Train model kfold 1**
* **Test videos 2 → Train model kfold 2**
* **Test videos 3 → Train model kfold 3**
* **Test videos 4 → Train model kfold 4**

The model is loaded from a checkpoint specified in:

```python
config.inference_model
```
Using mismatched data and models will lead to incorrect results.

#### Inputs:

Set all paths in `config.py`:

* `frames_dir`: video frames
* `embeddings_dir`: audio embeddings
* `audio_AEM_dir`: audio energy maps
* `videos_test_file`: list of test videos
* `inference_model`: pretrained model checkpoint

#### Run

```bash
python inference.py
```

The script loads a pretrained **AViSal360** model and applies it to a set of test videos. For each frame, it predicts a saliency map based on:

* RGB frames
* Audio Energy Maps (AEM)
* Audio embeddings

The output consists of predicted saliency maps saved as images, and optionally, videos generated from these predictions.

#### Notes

* Ensure all paths in `config.py` are correctly set before running.
* GPU is used automatically if available.

---
### Saliency Metrics Computation

The file `compute_metrics.py` evaluates predicted saliency maps against ground truth using standard saliency metrics.

This implementation is adapted from the toolbox of:

> E. David et al., *A Dataset of Head and Eye Movements for 360° Videos*, ACM MMSys 2018.

#### Supported Metrics

The script computes a set of commonly used metrics, including:

* **Fixation-based**: AUC-Judd, AUC-Borji, NSS
* **Saliency-based**: CC, SIM, KLD, EMD, MSE, MAE

The metrics to compute and their order are defined in:

```python
config.metrics_to_compute
```


#### Inputs

Set the following paths in `config.py`:

* `predicted_salmaps_path`: predicted saliency maps
* `gt_salmap_path`: ground-truth saliency maps
* `gt_fixations_file_path`: fixation data (CSV, optional but required for fixation-based metrics)
* `output_cvs_file_path`: output CSV file
* `salmaps_resolution`: expected resolution (width, height)
* `sampling_type`: sampling strategy for 360° data


##### 360° Sampling Strategies

To properly evaluate equirectangular saliency maps, the script supports:

* **Sphere sampling (`Sphere_N`)**
  Uniform sampling over the sphere

* **Sin weighting (`Sin`)**
  Latitude-based weighting to compensate equirectangular distortion

We used sin weighting for our results.

#### Output

A CSV file containing per-frame metrics:

```
Video | Frame | AUC_Judd | NSS | CC | ...
```

Each row corresponds to one frame of one video.

---

