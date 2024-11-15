# RMBG-2-Studio

An enhanced Pinokio app (with install files) built on [BRIA-RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

## Features

- Background Removal: Powered by BRIA-RMBG-2.0
- Drag-and-Drop Gallery: View your processed images and drag them directly from the gallery into the composition windows for background replacement and color grading
- Image Composition: Place and adjust processed images onto new backgrounds
- Color Grading: Adjust brightness, contrast, saturation, temperature, and tinting in the composition workspace
- Batch Processing: Process multiple images at once
- URL Support: Load images directly from URLs

## Pinokio UI: 
Tabbed UI:  Background Removal | Composition Workspace | Batch Removal
<p align="center">
  <a href="https://github.com/user-attachments/assets/d61a35eb-f426-4bd7-8a6b-f0f9ae6423d7" style="vertical-align: top; display: inline-block;">
    <img src="https://github.com/user-attachments/assets/d61a35eb-f426-4bd7-8a6b-f0f9ae6423d7" width="32%" alt="Background Removal">
  </a>
  <a href="https://github.com/user-attachments/assets/50d138c8-4729-4855-889c-afbd625f6e20" style="vertical-align: top; display: inline-block;">
    <img src="https://github.com/user-attachments/assets/50d138c8-4729-4855-889c-afbd625f6e20" width="32%" alt="Image Composition">
  </a>
  <a href="https://github.com/user-attachments/assets/df4b3ab7-6739-490e-ad29-50a77088ebd1" style="vertical-align: top; display: inline-block;">
    <img src="https://github.com/user-attachments/assets/df4b3ab7-6739-490e-ad29-50a77088ebd1" width="32%" alt="Batch Processing">
  </a>
</p>




### Credits & License
- Built on [BRIA-RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
- BRIA-RMBG-2.0 model is released under a Creative Commons license for non-commercial use only
- Commercial use of BRIA-RMBG-2.0 requires a commercial agreement with BRIA. [Contact BRIA](https://bria.ai/contact-us) for commercial licensing
- RMBG-2-Studio is free and open-source software, but this does not grant commercial usage rights to the underlying BRIA model




### Original BRIA-RMBG-2.0 README (Condensed)


# BRIA Background Removal v2.0 Model Card

RMBG v2.0 is our new state-of-the-art background removal model, designed to effectively separate foreground from background in a range of
categories and image types. This model has been trained on a carefully selected dataset, which includes:
general stock images, e-commerce, gaming, and advertising content, making it suitable for commercial use cases powering enterprise content creation at scale. 
The accuracy, efficiency, and versatility currently rival leading source-available models. 
It is ideal where content safety, legally licensed datasets, and bias mitigation are paramount. 

Developed by BRIA AI, RMBG v2.0 is available as a source-available model for non-commercial use. 

[CLICK HERE FOR A DEMO](https://huggingface.co/spaces/briaai/BRIA-RMBG-2.0)

## Model Details
#####
### Model Description

- **Developed by:** [BRIA AI](https://bria.ai/)
- **Model type:** Background Removal 
- **License:** [bria-rmbg-2.0](https://bria.ai/bria-huggingface-model-license-agreement/)
  - The model is released under a Creative Commons license for non-commercial use.
  - Commercial use is subject to a commercial agreement with BRIA. [Contact Us](https://bria.ai/contact-us) for more information. 

- **Model Description:** BRIA RMBG-2.0 is a dichotomous image segmentation model trained exclusively on a professional-grade dataset.
- **BRIA:** Resources for more information: [BRIA AI](https://bria.ai/)


### Architecture
RMBG-2.0 is developed on the [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) architecture enhanced with our proprietary dataset and training scheme. This training data significantly improves the model’s accuracy and effectiveness for background-removal task.<br>
If you use this model in your research, please cite:

```
@article{BiRefNet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}
```

BRIA's RMBG v2.0 model is ideal for applications where high-quality background removal is essential, particularly for content creators, e-commerce, and advertising. The model’s ability to handle various image types, including challenging ones with non-solid backgrounds, makes it a valuable asset for businesses focused on legally licensed, ethically sourced datasets.
