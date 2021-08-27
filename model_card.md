# Model Card - Census Bureau Income

## Model Details
* The Census Bureau classifier provided model for predict
income by social data information.
* Gradient Boosting Classifier
* Developed by Vyacheslav Khvan in 2021

## Intended Use
* Prediction task is to determine whether a person makes over 50K a year.
* Itended for real-time inference

## Factors
* Potential relevant factors include groups for gender and race.

## Metrics
* The model was evaluated using Fbeta, precision and recall.
* Pinned metrics measures for categorical features in dataset

## Training Data
* The data was obtained from the UCI Machine Learning Repository
* https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
* Train test split
* Compute model slice performance

## Quantative Analysis
<img src="https://github.com/vykhvan/ml-devops-ci-cd/blob/main/quantative-analysis.png" alt="drawing" width="800"/>

## Ethical Considerations
* Amer-indian Eskimo
 
## Caveats and Recommendations
* Low perfomance for Amer-indian Eskimo race. New more data for this group.
