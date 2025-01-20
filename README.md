# FaceRecognition
This repository presents a face recognition system trained using Siamese Neural Networks with two loss functions—contrastive and triplet—across six models. For each loss type, three models were developed, including one with pretrained weights (EfficientNet), to compare performance.

## Experimental Setup

### Dataset Configuration
- **Dataset**: CelebA
- **Number of Identities**: 1,000
- **Samples per Identity**: 6
- **Total Images**: 6,000
- **Image Size**: 128x128x3 / (224x224x3 for v3)

### Training Configuration
- **Batch Size**: 32
- **Epochs**: 100
- **Optimizer**: Adam
- **Data Split**: 
  - Training: 70%
  - Validation: 15%
  - Test: 15%

### Model-Specific Learning Rates
| Model | Learning Rate | Architecture Highlights |
|-------|---------------|------------------------|
| Contrastive V1 | 1e-3 | Basic CNN |
| Contrastive V2 | 5e-4 | Enhanced with residual connections |
| Contrastive V3 | 1e-4 | EfficientNetB0 backbone |
| Triplet V4 | 1e-5 | Basic architecture with LeakyReLU |
| Triplet V5 | 1e-5 | Enhanced with 300-dim embeddings |
| Triplet V6 | 1e-4 | EfficientNetB3 backbone |

## Implementation Details

### Training Strategy
- **Data Augmentation**:
  - Random horizontal flip
  - Random brightness adjustment (±20%)
  - Random contrast adjustment (±20%)
- **Regularization**:
  - Dropout (rates varying from 0.2 to 0.3)
  - Batch Normalization
  - L2 Normalization of embeddings
- **Early Stopping**: Patience of 10 epochs
- **Learning Rate Schedule**: Reduction on plateau
  - Monitor: Validation loss
  - Factor: 0.5
  - Patience: 5 epochs
  - Minimum LR: 1e-7

### Loss Functions
- **Contrastive Loss**: 
  ```python
  loss = y_true * squared_distance + (1 - y_true) * max(margin - distance, 0)²
  ```
- **Triplet Loss**:
  ```python
  loss = softplus(pos_dist - neg_dist + margin) + regularization
  ```

## Training Analysis

#### Loss Convergence Patterns

1. **Contrastive Loss Models**:
   - V1: Fast initial convergence, reaching validation loss of ~0.08 by epoch 25
   - V2: Steady convergence with lowest validation loss (~0.07) among contrastive models
   - V3: Most stable training curve with consistent validation loss around 0.08

2. **Triplet Loss Models**:
   - V4: Gradual improvement, final validation loss ~0.45
   - V5: Better convergence than V4, stabilizing at ~0.38 validation loss
   - V6: Best performer among triplet models with ~0.41 validation loss

#### Learning Rate Impact
- All models used learning rate reduction on plateau
- Initial learning rates varied by architecture:
  - Contrastive models: 1e-3 to 1e-4
  - Triplet models: 1e-4 to 1e-5

#### Training Stability
- Contrastive models showed more stable training curves
- Triplet networks had higher loss values but better final metrics
- Transfer learning models (V3 and V6) demonstrated consistent training patterns

## Results and Analysis

### Model Performance Comparison

#### Test Set Metrics
| Model Type | Version | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|------------|---------|-----------|------------|--------|-----|---------|---------|
| Contrastive | v1 | 81% | 79% | 82% | 81% | 88% | 86% |
| Contrastive | v2 | 81% | 81% | 80% | 80% | 89% | 87% |
| Contrastive | v3 (B0) | 80% | 76% | 87% | 81% | 87% | 83% |
| Triplet | v4 | 75% | 73% | 79% | 76% | 83% | 81% |
| Triplet | v5 | 80% | 75% | 89% | 81% | 88% | 86% |
| Triplet | v6 (B3) | 84% | 79% | 93% | 86% | 91% | 88% |

### Key Findings

1. **Best Overall Performance**: 
   - Triplet Network v6 (with EfficientNetB3 backbone) achieved the highest performance across most metrics:
     - Highest accuracy (84%)
     - Best recall (93%)
     - Highest F1 score (86%)
     - Best ROC AUC (91%)

2. **Architecture Comparison**:
   - Contrastive models (v1-v3) showed consistent performance around 80-81% accuracy
   - Triplet networks showed more variation but achieved higher peak performance
   - Transfer learning models (v3 and v6) demonstrated strong results

3. **Trade-offs**:
   - Contrastive networks showed better balance between precision and recall
   - Triplet networks, especially v6, excelled in recall but with slightly lower precision

### Visualization Guidelines

The project includes training history visualization capabilities through the `plots.ipynb` notebook:

1. **Training Curves Analysis**:
   ```python
   from plots import analyze_models
   
   # Define paths to training history JSON files
   model_json_dict = {
       'Contrastive_v1': 'path/to/contrastive_v1.json',
       'Contrastive_v2': 'path/to/contrastive_v2.json',
       'Contrastive_v3': 'path/to/contrastive_v3.json',
       'Triplet_v4': 'path/to/triplet_v4.json',
       'Triplet_v5': 'path/to/triplet_v5.json',
       'Triplet_v6': 'path/to/triplet_v6.json'
   }
   
   # Generate comparison plots
   analyze_models(model_json_dict)
   ```

2. **Performance Metrics**:
   - `evaluate_contrastive.ipynb` or `evaluate_triplet.ipynb` to evaluate on test data for:
     - ROC curves with AUC scores
     - Precision-Recall curves
     - Similarity score distributions

3. **Real-world Testing**:
   - Use `test_on_images.ipynb` for:
     - Face verification demos
     - Face identification in galleries
     - Visual similarity analysis


## Dataset License and Usage

This project uses the [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for face recognition research. 

### Agreement
The CelebA dataset is available for non-commercial research purposes only. By using this dataset, you agree:
- Not to reproduce, duplicate, copy, sell, trade, resell, or exploit any part of the images or derived data for commercial purposes.
- Not to further copy, publish, or distribute any portion of the dataset, except for internal use at a single site within the same organization.
- The MMLAB reserves the right to terminate access at any time.

All images are sourced from the Internet and are not the property of MMLAB. MMLAB is not responsible for their content or meaning.

### Citation
```bibtex
@inproceedings{liu2015faceattributes,
  title={Deep Learning Face Attributes in the Wild},
  author={Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={Proceedings of International Conference on Computer Vision (ICCV)},
  month={December},
  year={2015}
}

