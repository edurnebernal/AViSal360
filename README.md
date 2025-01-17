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

## Proposed Audio Energy Maps
Details about our proposed audio energy maps are provided in the [supplementary material](https://nas-graphics.unizar.es/s/8WcJBwaJM2PNPE2). You can access the code for generating the AEMs at our GitHub repository: [AmbisonicPowermapTest](https://github.com/R3Ngfx/AmbisonicPowermapTest).
