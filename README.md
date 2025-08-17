🩺 Breast Cancer Prediction Model – Performance Analysis

This project leverages LightGBM (LGBMClassifier) to predict whether a tumor is malignant or benign using the Breast Cancer dataset. The model was fine-tuned using RandomizedSearchCV, achieving exceptional results across multiple performance metrics.

📊 Model Performance

Accuracy: 96%

Precision:

Class 0 (Benign): 1.00

Class 1 (Malignant): 0.95

Recall:

Class 0 (Benign): 0.91

Class 1 (Malignant): 1.00

F1 Score:

Class 0 (Benign): 0.95

Class 1 (Malignant): 0.97

🔹 These results demonstrate that the model can effectively detect malignant tumors without missing any cases (recall = 100%), while also minimizing false positives.

🖼️ Visualizations
Confusion Matrix
<img width="640" height="480" alt="confusion_matrix_bcp" src="https://github.com/user-attachments/assets/d57ad1dc-667d-46f9-b2ed-e8d3509ccff5" />


ROC Curve
<img width="640" height="480" alt="roc_curve_bcp" src="https://github.com/user-attachments/assets/cd6289ba-6845-46ba-8b2b-05804902f837" />


⚙️ Key Features

Algorithm: LGBMClassifier with tuned hyperparameters (max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=200).

Evaluation: Precision, Recall, F1 Score, Accuracy, ROC Curve, Confusion Matrix.

Dataset: Breast Cancer dataset (binary classification).

Result: Achieved 96% overall accuracy with excellent recall on malignant cases (100%), making the model highly reliable for medical screening tasks.

🔥 This project showcases how ML + healthcare can work together to assist in early breast cancer detection.
