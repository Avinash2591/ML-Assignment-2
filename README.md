### ML Assignment 2 ###



### a. Problem statement

The core objective of this project is to develop an intelligent classification system capable of distinguishing between "High-End" and "Standard" luxury vehicles within the BMW product lineup (2012-2025). Using a dataset containing technical specifications such as engine displacement, horsepower, and torque, the project aims to identify the underlying patterns that correlate vehicle performance and features with market pricing tiers.

To achieve this, the project implements a comparative analysis of six distinct machine learning algorithms: Logistic Regression, Decision Tree, k-Nearest Neighbors (kNN), Naive Bayes, Random Forest, and XGBoost. The technical goals of the assignment include:

**Binary Classification**: Converting raw MSRP (Manufacturer's Suggested Retail Price) data into a binary target based on a median price threshold ($55,695).

**Feature Engineering**: Processing and scaling 12+ multi-modal features, including categorical variables like "Body Type" and numerical variables like "0-60 mph" acceleration times.

**Performance Evaluation**: Benchmarking models using 6 mandatory metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and the Matthews Correlation Coefficient (MCC).

**Operational Deployment**: Building a user-centric web interface using Streamlit to allow stakeholders to upload car data for real-time inference and performance visualization.

By deploying this solution to the Streamlit Community Cloud, the project demonstrates a complete "Model-to-Product" pipeline suitable for modern data science applications.taset.


### b. Dataset Description

The dataset comprises technical and market specifications for 157 BMW vehicle models spanning the years 2012 to 2025. It contains 18 initial columns, from which 12 primary features were selected to train the classification models.
For the purpose of this assignment, a binary target variable (Target) was engineered by splitting the Manufacturer's Suggested Retail Price (MSRP_USD) at its median value of $55,695. Models were trained to predict whether a car belongs to the "High-End" tier (MSRP > Median) or the "Standard" tier (MSRP ≤ Median).

Feature Breakdown:

**Model & Series**: The specific name and vehicle line (e.g., 3 Series, X5).

**Year**: The production year (2012–2025).

**Engine & Transmission**: Technical specifications including Engine_Type, Displacement_L, Cylinders, and Transmission type.

**Performance Metrics**: Key indicators such as Horsepower, Torque_lb_ft, 0_60_mph_sec acceleration, and Top_Speed_mph.

**Efficiency**: Fuel_Economy_City_mpg and Fuel_Economy_Highway_mpg.

**Dimensions**: Body_Type (e.g., Sedan, SUV, Coupe) and Seating_Capacity.

**Drivetrain**: Wheel drive configuration (RWD, AWD).


## c. Comparison Table with the evaluation metrics calculated for all the 6 models as below:


| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
| :---                     | :---     | :---  | :---      | :---   | :---  | :---  |
| Logistic Regression      | 0.936    | 0.976 | 0.947     | 0.923  | 0.935 | 0.873 |
| Decision Tree            | 0.994    | 0.994 | 0.987     | 1.000  | 0.994 | 0.987 |
| kNN                      | 0.898    | 0.967 | 0.908     | 0.885  | 0.896 | 0.796 |
| Naive Bayes              | 0.866    | 0.950 | 0.852     | 0.885  | 0.868 | 0.733 |
| Random Forest (Ensemble) | 0.994    | 1.000 | 0.987     | 1.000  | 0.994 | 0.987 |
| XGBoost (Ensemble)       | 0.892    | 0.953 | 1.000     | 0.782  | 0.878 | 0.802 |


## d. Observations on the performance of each model on the chosen dataset.

| ML Model Name           | Observation about model performance |
| :---                    | :---                                |
| **Logistic Regression** | Performed strongly with high AUC (0.976), indicating that technical specifications like Horsepower and Engine Size have a robust linear relationship with vehicle price tiers.|
| **Decision Tree**       | Achieved near-perfect metrics with 1.000 Recall, effectively identifying every "High-End" vehicle. This suggests clear, distinct decision boundaries in the performance data.|
| **kNN**                 | Delivered reliable results (0.898 Accuracy) but was sensitive to feature scaling, as car prices are influenced by the complex proximity of technical specs across different series.|
| **Naive Bayes**         | The lowest overall performer (0.866 Accuracy), likely due to the inherent correlation between features like Displacement and Horsepower, which violates the model's independence assumption.|
| **Random Forest**       | Tied for the highest accuracy and achieved a perfect AUC (1.000). The bagging technique significantly stabilized predictions by reducing variance found in single decision trees.|
| **XGBoost**             | Showed unique behavior with perfect Precision (1.000) but lower Recall (0.782). It was highly conservative, ensuring that any car labeled "High-End" truly belonged in that category.|
