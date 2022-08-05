# AAAI Submission Experiments and Figures
Experiments listed in order of priority, with datasets. Figures and data are in the results/ folder.

### Bias
- [x] Toy example with regression
- [x] MNIST highest-bias images
- [x] VAE + Face Dataset (skin types with biases and accuracies done, more visualizations?)
- [ ] Feature drop experiment
- [ ] Low priority: driving dataset?

### Aleatoric
- [x] Toy example with regression
- [x] MNIST mislabeled images
- [ ] Incorrect segmentations with Kitti dataset

### Epistemic
- [ ] Calibration curves with the NYU Depth dataset (for all epistemic methods)
- [ ] OOD/Anomaly detection with NYU Depth
- [ ] Low priority: table with RMSE, NLL

### Composability
- [x] Toy example with regression
- [ ] Ensembling the MVE metric (both model + MVE, as well as just MVE) (which dataset?)
- [ ] Ensembling dropout + MVE with dropout (which dataset?)

### Misc