# Weights Directory

This folder stores model checkpoint files for the MobileNetV3 Mobile Detection system.

## Checkpoint Files
- `best_mobile_detector.pth`: The PyTorch `state_dict` of the trained model with the highest validation accuracy.
- `training_history.json`: Epoch-by-epoch loss and accuracy metrics.
- `training_curves.png`: Visual loss and accuracy curve graphs.

## How to Obtain Weights
1. **Train Locally or on Colab**:
   Run:
   ```bash
   python main.py train --data-dir path/to/binary_dataset
   ```
   The best checkpoint will automatically be saved to `best_mobile_detector.pth`.

2. **From Colab**:
   If you ran `FYP_Mobile_Detection (1).ipynb` on Google Colab, download `best_mobile_detector.pth` from Colab files and place it in this directory.

3. **Fallback Mode**:
   If no weights file is provided, the system initializes MobileNetV3-Large with default ImageNet pretrained backbone weights so the code remains functional for testing pipeline flow.
