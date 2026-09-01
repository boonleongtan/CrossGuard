# Robust Detection of AI-Generated Images Under Real-World Transformations

## 5.1 Background

Generative AI tools are making it easier than ever to create highly realistic synthetic images at scale. This creates new risks for online platforms, including misinformation, impersonation, fraud, and reduced trust in digital content. In practice, detection becomes even harder after images are compressed, cropped, reposted, or lightly edited, so robust methods matter more than lab-only accuracy.

## 5.2 Problem Statement

We want participants to build a prototype that can distinguish AI-generated images from authentic images with strong robustness under realistic post-processing and redistribution scenarios. The goal is not only to achieve good detection performance on clean data, but also to maintain accuracy after transformations such as blur, compression, color adjustment, cropping, or rescaling. Solutions should present a clear technical approach, an evaluation strategy, and thoughtful discussion of trade-offs such as robustness, generalisation, and false positives.

**Note: We consider robustness against a subset of the following augmentataions.**

| Transform        | Parameters                        | Real-World Analog                 |
| ---------------- | --------------------------------- | --------------------------------- |
| JPEG Compression | quality = 90, 70, 50, 30          | Social-media re-encode, messaging |
| Gaussian Blur    | kernel σ = 0.5, 1.0, 2.0         | Out-of-focus                      |
| Resize           | scale 0.5× / 0.25× then upscale | Thumbnail generation              |
| Gaussian Noise   | σ = 0.02, 0.05, 0.10             | Low-light sensor noise            |
| Color Jitter     | brightness/contrast/sat. ±20%    | Filter apps, auto-enhance         |
| Center Crop      | crop 80%                          | Profile-picture cropping, framing |

## 5.3 Constraints & Scope

| Category            | Constraints & Scope Details                                                                                                                                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| In scope            | Image-level AIGC detection, robustness to common image transformations, feature engineering, model design, evaluation design, error analysis, and explainability ideas                                                                                              |
| Out of scope        | Full production deployment, platform-wide moderation systems, and non-image modalities such as video or audio                                                                                                                                                       |
| Limits              | Assume a hackathon-scale prototype, limited compute, and no access to internal production systems. Teams should optimise for a convincing proof of concept rather than a production-grade service.**Note: Participants must use models with <2B parameters.** |
| Allowed assumptions | Teams may use public or properly licensed datasets, create their own transformed test cases, and make reasonable assumptions about deployment context as long as those assumptions are stated clearly.                                                              |

## 5.4 Available Resources & Data

- Public or properly licensed image datasets for AIGC detection and image forensics.
- Self-created transformed samples using operations such as blur, compression, cropping, color adjustment, or rescaling.
- Public documentation for relevant machine learning and computer vision libraries.
- Datasets:
  - https://huggingface.co/datasets/saberzl/SID_Set
  - https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
  - https://modelscope.cn/datasets/hy2628982280/WildFake/summary
  - For this modelscope dataset, please translate it via the translation button before use.

### Validation Dataset (for Demonstration Purposes Only)

We choose **a subset of WildFake** for participants to demonstrate their models' performance and track iterative improvements. This dataset serves only as a reference benchmark and will not contribute to the final score. **Do not use the following data during training.** Specifically:

| Dataset  |                  | # Num |
| -------- | ---------------- | ----- |
| Non-AIGC | COCO val2017     | 4998  |
| AIGC     | DALL·E Advanced | 8843  |

## 5.5 Expected Deliverables

### 1. Written Project Description (via Devpost)

- Provide a clear written description of your project that includes:
  - How your solution addresses the problem statement
  - Development tools used (e.g. VSCode, Colab, Jupyter)
  - Models or APIs used
  - Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)
  - Datasets and assets used

### 2. Public Code/GitHub Repository

- Submit a link to a public Code/GitHub repository containing:
  - Well-structured, commented code covering all components of your solution
  - A script that takes an image directory as input and outputs a confidence score for each image, indicating the likelihood that it is AIGC-generated. The output should be a JSON file containing `image_path` and `pred` for each image.
  - A README file that includes:
    - Project overview
    - Setup and installation instructions
    - Steps to reproduce your results
    - A brief reflection on your solution's limitations and what you would improve given more time
    - Team member contributions (if applicable, i.e. team participants, non-solo participants)

### 3. Demo Video

- Submit a short video that:
  - Demonstrates your solution working end-to-end (e.g. inference results, dashboard, model predictions)
  - Is uploaded to YouTube and set to public visibility
  - Is linked in your Devpost description
  - Does not include third-party trademarks or copyrighted content without permission

### 4. Robustness Evaluation Summary

- Include a compact table or visual summary comparing performance on clean images versus transformed images.

### 5. Error Analysis Note

- Highlight representative false positives, false negatives, and any trade-offs in the proposed approach.

## 5.6 Judging Criteria

| Judging Criteria                       | Definition                                                                                                                                                                                                                                                              | Weight |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| **Technical Execution**          | The solution demonstrates strong engineering fundamentals, such as well-structured code, thoughtful architecture, and effective use of APIs or models. The demo runs reliably, and the technical complexity reflects deliberate, capable decision-making.               | 35%    |
| **Innovation & Problem Insight** | The project demonstrates originality in both idea and approach. It stands out for the sharpness of its problem understanding: how clearly the team has framed the challenge, why it matters, and how directly the solution addresses it.                                | 20%    |
| **Impact & Relevance**           | The project has clear potential to deliver value to real users or stakeholders, with meaningful reach, tangible benefit, and relevance that goes beyond solving for the hackathon prompt alone.                                                                         | 20%    |
| **Feasibility & Practicality**   | The solution is realistic and buildable beyond a prototype. The approach is technically and operationally sustainable: resource usage is proportionate, the architecture holds under real-world conditions, and the implementation is grounded rather than speculative. | 15%    |
| **Presentation & Communication** | The team communicates their work with clarity. [Final Event Only]: The pitch tells a coherent story; from problem to solution to potential, and the team is able to respond to questions with depth, demonstrating genuine understanding of their own project.          | 10%    |
