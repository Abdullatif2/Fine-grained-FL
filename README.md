# Fine-Grained Federated Edge Learning (FEEL)
This codebase implements the novel fine-grained training algorithm and resource optimization techniques proposed in our paper for Federated Edge Learning (FEEL) titled "Fine-Grained Data Selection for Improved Energy Efficiency of Federated Edge Learning"


This repository contains the implementation of the methodologies presented in the paper:  
**"Fine-Grained Data Selection for Improved Energy Efficiency of Federated Edge Learning"**  
Published in the *IEEE Transactions on Network Science and Engineering*, 2022.  
[Link to Paper](https://arxiv.org/abs/2106.12561)

## Overview

Federated Edge Learning (FEEL) enables edge devices to collaboratively train machine learning models without sharing raw data, thereby preserving privacy. However, the energy consumption associated with local training and communication can be substantial, especially for energy-constrained devices. This repository addresses this challenge by implementing a fine-grained data selection algorithm that optimizes energy efficiency without compromising model performance.

The key contributions include:

- **Fine-Grained Data Selection Algorithm**: Selects the most relevant training samples to reduce computational load and energy consumption.
- **Threshold-Based Data Exclusion .....



## Dependencies

Ensure that the following Python libraries are installed:

- NumPy
- Pandas
- Scikit-learn
- PyTorch 
- Matplotlib


## Citation

If you find this repository or the corresponding paper useful in your research, please cite:

```bibtex
@article{albaseer2022fine,
  author={Albaseer, Abdullatif and Abdallah, Mohamed and Al-Fuqaha, Ala and Erbad, Aiman},
  journal={IEEE Transactions on Network Science and Engineering}, 
  title={Fine-Grained Data Selection for Improved Energy Efficiency of Federated Edge Learning}, 
  year={2022},
  volume={9},
  number={5},
  pages={3258-3271},
  doi={10.1109/TNSE.2021.3100805}}
}
