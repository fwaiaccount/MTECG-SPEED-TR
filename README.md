# MTECG-SPEED-TR




## 🔹 Released Components
We have made the following components publicly available:
- **Final model architecture**




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


## 🙏 Acknowledgements & References

This work builds upon prior contributions in **vision transformers** and **masked autoencoders**. Some components were inspired by the design of DeiT, BEiT, and MAE, but have been adapted and extended for ECG-specific tasks.  

### Open-source repositories
- [DeiT (Data-efficient Image Transformers)](https://github.com/facebookresearch/deit)  
- [BEiT (Bidirectional Encoder representation from Image Transformers)](https://github.com/microsoft/unilm/tree/master/beit)  
- [MAE (Masked Autoencoders)](https://github.com/facebookresearch/mae)  

### Related research paper
- Zhou, Y., Diao, X., Huo, Y., Liu, Y., Sun, Z., Fan, X., & Zhao, W. (2025). *Enhancing automatic multilabel diagnosis of electrocardiogram signals: A masked transformer approach*. **Computers in Biology and Medicine, 196**, 110674. [DOI link](https://doi.org/10.1016/j.compbiomed.2025.110674)
