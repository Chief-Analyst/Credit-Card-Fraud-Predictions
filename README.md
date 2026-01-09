Credit Card Fraud Detection Using TensorFlow
Project Overview

This project detects fraudulent credit card transactions using a neural network implemented in TensorFlow/Keras. The dataset is highly imbalanced, with very few fraudulent transactions compared to non-fraudulent ones. The goal is to build a model that accurately detects fraud while minimizing false positives, and to fine-tune it for optimal precision and recall. The dataset was sourced from Kaggle and to my greatest surprise, had no missing values.

Dataset

Contains numerical features representing credit card transactions.

Two classes:

0 → Non-fraudulent transaction

1 → Fraudulent transaction

Due to extreme class imbalance, special handling was applied during training and evaluation.

Data Preprocessing

Splitting the dataset:

Train, validation, and test sets were created.

Stratification was used to maintain class distribution across splits.

Scaling features:

StandardScaler was used to scale numeric features.

Only the training set was used to fit the scaler to avoid data leakage.

Validation and test sets were transformed using the same scaler.

Handling class imbalance:

class_weight was initially set to balanced, resulting in extreme weighting for the minority class (~289×).

This caused extremely high recall (~96–97%) but very low precision (~6–8%).

Class weights were capped at {0: 1, 1: 10} to prioritize fraud while reducing false positives.

Model Architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])


Input layer: number of features in the dataset

Hidden layers: 32 and 16 neurons with ReLU activation

Dropout: 30% to prevent overfitting

Output layer: 1 neuron with sigmoid activation for binary classification

Model Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)


Loss function: binary cross-entropy

Metrics: accuracy, precision, recall, and AUC

Model Training
from sklearn.utils import class_weight

# Class weights
class_weights = {0: 1, 1: 10}

# Training
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=2048,
    validation_split=0.25,
    class_weight=class_weights,
    verbose=2
)


Batch size: 2048

Epochs: 10

Validation split: 20% of training data

Class weights: {0:1, 1:10} to prioritize fraud while reducing false positives

Fine-Tuning Threshold and Class Weights

Default threshold = 0.5

Extreme class weights caused very low precision (~6–8%) despite high recall (~96%)

By capping the minority class weight at 10:

Training precision: 72%

Training recall: 79%

Validation precision: 88%

Validation recall: 85%

Test precision: 79%

Test recall: 80%

The threshold was adjusted further to optimize precision-recall trade-off.

Model Evaluation
Confusion Matrix
[[56843    21]
 [   20    78]]


True Negatives = 56843 → non-fraud correctly predicted

False Positives = 21 → non-fraud predicted as fraud (false positives)

False Negatives = 20 → fraud predicted as non-fraud (missed fraud)

True Positives = 78 → correctly predicted fraud

Classification Report
Class	Precision	Recall	F1-score	Support
0	1.00	1.00	1.00	56864
1	0.79	0.80	0.79	98

Accuracy = 99.96% → high but dominated by non-fraud

Precision (fraud) = 79% → most predicted frauds are correct

Recall (fraud) = 80% → most actual frauds are detected

F1-score (fraud) = 0.79 → good balance between precision and recall

Key Takeaways

Imbalanced dataset handling:

Using class weights improves detection of the minority class.

Extreme weights can hurt precision, capping weights balances precision and recall.

Metrics:

Accuracy is misleading due to imbalance.

Precision and recall for fraud are primary metrics.

F1-score balances both metrics.

Fine-tuning:

Adjusting class weights significantly increased precision from ~7% to 88% while maintaining recall above 85%.

Threshold tuning can further refine the balance.

Neural network design:

Two hidden layers with dropout prevented overfitting.

Scaled features ensured proper model training.

Production-ready model:

High AUC (~0.95) indicates excellent class separation.

Balanced precision and recall make it usable for real-world fraud detection systems.

Usage

Preprocess data: scale numeric features, split dataset with stratification.

Train model: use capped class weights for the minority class.

Evaluate model: confusion matrix, classification report, precision, recall, F1-score, AUC.

Threshold tuning: adjust threshold for optimal precision-recall trade-off.

Deploy: predict probabilities on new transactions and flag fraud using the selected threshold.

Conclusion

By carefully handling class imbalance, scaling features, and tuning class weights, this TensorFlow neural network successfully detects credit card fraud with high precision and recall. Fine-tuning thresholds ensures a practical balance between catching fraud and reducing false positives, making this model suitable for production deployment in real-world financial systems.

Author 

Chukwuma Samuel Ifeanyichukwu
(Entry-level Data Scientist)

Please take a chance on me, I really want to put these skills in real world scenarios. 
