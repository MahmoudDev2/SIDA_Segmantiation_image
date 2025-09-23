# Research on Forgery Localization Models (Mask Prediction)

As requested, here is a list of alternative models and projects that are designed not only to detect fake images but also to **localize** the manipulated regions, often by producing a mask. These models are generally more complex than the simple classification model used in this Flask application.

## 1. LAA-Net

- **Paper:** Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection
- **GitHub Repository:** [https://github.com/10Ring/LAA-Net](https://github.com/10Ring/LAA-Net)
- **Description:** This model uses an "attention" mechanism to focus on localized artifacts. It's designed to be robust to varying image quality and generalizes well. It produces a mask highlighting the forged regions.

## 2. HiFi-IFDL

- **Paper:** Hierarchical Fine-Grained Image Forgery Detection and Localization
- **GitHub Repository:** [https://github.com/CHELSEA234/HiFi_IFDL](https://github.com/CHELSEA234/HiFi_IFDL)
- **Description:** This approach uses a hierarchical method to detect forgery at different scales, making it effective for various types of manipulations. It provides both a classification (real/fake) and a localization mask.

## 3. ST-DDL

- **Paper:** Exploring Spatial-Temporal Features for Deepfake Detection and Localization
- **GitHub Repository:** [https://github.com/HighwayWu/ST-DDL](https://github.com/HighwayWu/ST-DDL)
- **Description:** This model is specifically designed for **videos**. It analyzes features across both space (the image frame) and time (between frames) to detect and localize deepfakes. If your interest is in video, this is a relevant project.

These repositories can serve as a starting point for exploring more advanced forgery localization techniques.
