# Model Card - Census Bureau Income

## Model Details
* The Census Bureau classifier model for prediction task is to 
determine whether a person makes over 50K a year.
* Gradient Boosting Classifier.
* Developed by Vyacheslav Khvan in 2021.

## Intended Use
* Prediction task is to determine whether a person makes over 50K a year.
* Itended for real-time inference.

## Factors
* Potential relevant factors include groups for gender and race.

## Metrics
* Evaluation metrics include precision, recall and fbeta.
* Together, these metrics provide values for different errors.
* Pinned metrics measures for categorical features in dataset.
* Precision is 0.8, Recall is 0.63, Fbeta is 0.71. 

## Training Data
* The data was obtained from the UCI Machine Learning Repository.
* https://archive.ics.uci.edu/ml/datasets/census+income.
* Census income data, training data split.

## Evaluation Data
* Census income data, test data split.

## Quantative Analysis
<img src="https://github.com/vykhvan/ml-devops-ci-cd/blob/main/images/quantative-analysis.png" width="800"/>

## Ethical Considerations
* Race group Amer-indian Eskimo. 
 
## Caveats and Recommendations
* Low perfomance for Amer-indian Eskimo race.
* Use more data for training.
* Use more complex hyperparameters optimization. 
