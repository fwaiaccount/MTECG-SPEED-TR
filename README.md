# SPEED-TR: Self-Distilled and Pre-trained Transformer model for Enhanced ECG Detection of Tricuspid Regurgitation

This repository provides the **official implementation** of the paper:

**"SPEED-TR: Self-Distilled and Pre-trained Transformer model for Enhanced ECG Detection of Tricuspid Regurgitation with Multi-Center Validation and Comprehensive Risk Factor Analysis" (2025+)**

---

## 🔹 Released Components
We have made the following components publicly available:
- **Final model architecture**
- **Self-distillation code** proposed in the paper


## 🔹 File Structure & Usage
This section provides a description of the main components and their file locations. You can use the code as follows:

### 1. **Model Architecture**
- **File Location:** `./MTECG_arch/models`
- **Description:** This directory contains the final model architecture used in the SPEED-TR system. It implements the Transformer model tailored for ECG detection tasks.
- **Usage:** To instantiate the model and perform a forward pass, use the following code:

```python
import torch
import MTECG_arch.models.et_family as et_family

# Initialize model architecture
model_names = 'ti_12_25'  # Model name
model = et_family.__dict__[model_names](
    num_classes=1,  # Number of output classes (TR detection)
)

# Example input: 12-lead ECG signal with a sampling rate of 500Hz and 10 seconds duration
X = torch.randn(1, 1, 12, 5000)  # Shape: (batch_size, 1, leads, time points)

# Forward pass example
y = model(X)
```
### 2. **Self-Distillation Code**
- **File Location:** `self_distillation_loss.py`
- **Description:** This file details the knowledge distillation loss function used by SPEED-TR. Our knowledge distillation method builds upon [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) and the [PyTorch Tutorial: Knowledge Distillation](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html), with modifications to the temperature and loss weighting tailored for our task.
- **Usage:** To apply this self distillation loss, please use the following code:
```python
import torch
import torch.nn
import models.et_family  as et_family
from self_distillation_loss import distillation_loss
model_names = 'ti_12_25'  # Model name

teacher_model = et_family.__dict__[model_names](  
    num_classes=1,# Number of output classes (TR detection)
)
checkpoint =torch.load('./teacher_model_checkpoint.pth', map_location='cpu') #Load teacher model parameters
checkpoint_model = checkpoint['model']
msg = teacher_model.load_state_dict(checkpoint_model, strict=False)

student_model = et_family.__dict__[model_names](
    num_classes=1,
)

# Example input: 12-lead ECG signal with a sampling rate of 500Hz and 10 seconds duration
X = torch.randn(1,1,12,5000)  # Shape: (batch_size, channels, leads, samples)
targets=torch.randn(1,1)  # Shape: (batch_size, labels)

with torch.no_grad():
    teacher_logits = teacher_model(X)  #Forward pass to get the teacher model output
student_logits = student_model(X)#Forward pass to get the student model output

 #Calculate the loss
loss=distillation_loss(student_logits, teacher_logits, targets, 
                      temperature=2.0, distillation_weight=0.7, criterion=torch.nn.BCEWithLogitsLoss()) 
```

## 🙏 Acknowledgements & References

This work builds upon prior contributions in **vision transformers** and **masked autoencoders**. Some components were inspired by the design of DeiT, BEiT, and MAE, but have been adapted and extended for ECG-specific tasks.  

### Open-source repositories
- [DeiT (Data-efficient Image Transformers)](https://github.com/facebookresearch/deit)  
- [BEiT (Bidirectional Encoder representation from Image Transformers)](https://github.com/microsoft/unilm/tree/master/beit)  
- [MAE (Masked Autoencoders)](https://github.com/facebookresearch/mae)  

### Related research paper
- Zhou, Y., Diao, X., Huo, Y., Liu, Y., Sun, Z., Fan, X., & Zhao, W. (2025). *Enhancing automatic multilabel diagnosis of electrocardiogram signals: A masked transformer approach*. **Computers in Biology and Medicine, 196**, 110674. [DOI link](https://doi.org/10.1016/j.compbiomed.2025.110674)
-  Hinton, O. Vinyals, and J. Dean. (2015). *Distilling the knowledge in a neural network*. arXiv preprint arXiv:1503.02531 [DOI link](https://arxiv.org/abs/1503.02531)
