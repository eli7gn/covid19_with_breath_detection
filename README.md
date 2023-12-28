# COVID19 Audio Classifier for Coswara Breath Dataset

## Overview
This project presents an audio classification system designed to identify 'positive' and 'negative' categories in the Coswara Breath Dataset. Utilizing a range of machine learning algorithms, the classifier demonstrates significant efficacy, particularly in maximizing recall to ensure all positive cases are identified.

## Dataset
The Coswara Breath Dataset comprises audio recordings categorized into 'positive' and 'negative' groups based on specific criteria. This dataset is instrumental in developing models that can distinguish between these two categories effectively.

## Algorithms Employed
The project harnesses the power of various machine learning algorithms, each offering unique strengths:

1. **XGBoost**: Known for its speed and performance, XGBoost is a gradient boosting framework that provides an efficient and effective method for classification tasks.
2. **Random Forest**: This ensemble learning method constructs a multitude of decision trees during training, outputting the mode of their classifications. It's known for its accuracy and ability to run efficiently on large datasets.
3. **Gradient Boosting**: Another ensemble technique that builds models sequentially, with each new model correcting the errors of its predecessors.
4. **KNeighbors**: An instance-based learning method that classifies cases based on a majority vote of their nearest neighbors.
5. **Gaussian Naive Bayes**: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between features.
6. **Combined Approach**: This approach aggregates predictions from all models, marking a case as positive if any model predicts it as such. It's particularly useful for maximizing recall.

## Performance Metrics
The models were evaluated based on accuracy, precision, and recall. Notably, the combined approach achieved a recall of 0.89, indicating its effectiveness in identifying positive cases:

- **Combined Model Recall**: 0.89

## Code Usage
The project is encapsulated in the `AudioClassifier` class, handling feature extraction, model training, and evaluation. Users can easily initialize the class and call its methods to process the dataset and evaluate the models.

## Results
Each model's performance, including the combined model, is summarized as follows:

- **XGBoost**: Accuracy: 0.89, Precision: 0.60, Recall: 0.33
- **Random Forest**: Accuracy: 0.93, Precision: 0.83, Recall: 0.56
- **Gradient Boosting**: Accuracy: 0.94, Precision: 0.78, Recall: 0.78
- **KNeighbors**: Accuracy: 0.93, Precision: 1.00, Recall: 0.44
- **Gaussian Naive Bayes**: Accuracy: 0.90, Precision: 0.58, Recall: 0.78
- **Combined Models**: Accuracy: 0.89, Precision: 0.53, Recall: 0.89

Confusion matrices for each model are also provided for a detailed breakdown of true positives, false positives, true negatives, and false negatives.

---
