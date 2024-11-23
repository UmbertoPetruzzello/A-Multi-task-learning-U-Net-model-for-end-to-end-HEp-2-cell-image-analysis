
# A Multi-task learning U-Net model for end-to-end HEp-2 cell image analysis

This repository contains the implementation of a multi-task learning framework designed for **end-to-end HEp-2 cell image analysis**. The framework addresses three critical tasks in HEp-2 image analysis:

1. **Intensity Classification**: Predicting the intensity of fluorescence in HEp-2 cell images.
2. **Specimen Segmentation**: Accurate segmentation of HEp-2 cells.
3. **ANA Pattern Classification**: Identifying antinuclear antibody (ANA) patterns.

These tasks are essential for building comprehensive systems for automated HEp-2 cell analysis.

## Features

- **Multi-Task Learning**: The framework integrates three tasks into a unified model, leveraging shared learning to improve performance across tasks.
- **Customizable Training**: Modify hyperparameters and enable data augmentation or preprocessing via command-line arguments.
- **Comprehensive Loss Function**: Combines Dice Loss, Cross-Entropy Loss, and Binary Cross-Entropy Loss for effective optimization.
- **End-to-End Solution**: Supports image input through to final predictions, including segmentation masks, intensity values, and pattern classifications.

## Project Structure

- **`cross_validation.py`**: Main script for training, validation, and evaluation. Includes utilities for dataset handling, model training, and performance metrics.
- **`model.py`**: Implementation of the U-Net architecture with extensions for intensity and pattern classification.
- **`loss.py`**: Definition of the custom multi-task loss function, combining segmentation, classification, and regression objectives.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/UmbertoPetruzzello/A-Multi-task-learning-U-Net-model-for-end-to-end-HEp-2-cell-image-analysis.git
   ```

2. Prepare your dataset:
   - Organize your dataset into the required structure.
   - Update dataset paths in `cross_validation.py`.

## Usage

### Training and Validation

Run the `cross_validation.py` script with customizable parameters:
```bash
python cross_validation.py --da True --pp True --dim_x 384 --dim_y 384
```

**Available Arguments**:
- `--da`: Enable or disable data augmentation (default: `False`).
- `--pp`: Enable or disable preprocessing (default: `False`).
- `--dim_x`, `--dim_y`: Dimensions of the input patches (default: `384x384`).

### Model Outputs

The model generates three outputs:
1. **Segmentation mask** for cell boundaries.
2. **Pattern classification** probabilities for ANA patterns.
3. **Intensity classification** values.

### Evaluation

- Metrics such as Dice Score, Accuracy, and Confusion Matrix are automatically calculated during testing.
- Outputs are saved in specified directories for further analysis.

## Results

This framework has been validated on HEp-2 cell datasets, achieving competitive performance across all tasks. Quantitative results and visualizations of segmentation and classification are included in the repository's output directories.

## Citation

If you use this code in your research, please cite the associated paper:

@article{MTL_HEP2_2024, 
title = {A Multi-task learning U-Net model for end-to-end HEp-2 cell image analysis},
journal = {Artificial Intelligence in Medicine},
pages = {103031},
year = {2024},
issn = {0933-3657},
doi = {https://doi.org/10.1016/j.artmed.2024.103031},
url = {https://www.sciencedirect.com/science/article/pii/S0933365724002732},
author = {Gennaro Percannella and Umberto Petruzzello and Francesco Tortorella and Mario Vento}
}


## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). You are free to use, modify, and distribute this software under the following terms:

Freedom to Use: You may use this software for any purpose.
Freedom to Modify: You can modify the software to fit your needs, provided that the source code remains available.
Freedom to Share: You may redistribute the software, but it must remain licensed under the GPL-3.0.
Freedom to Share Improvements: If you distribute a modified version, you must also distribute the source code and apply the same GPL-3.0 license.
For more details, refer to the GPL-3.0 License.