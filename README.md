# Data Preprocessing in Machine Learning 

Data Preprocessing is the process of cleaning and preparing raw data before training a machine learning model.

Think of it like washing and cutting vegetables before cooking. Good preprocessing helps models learn better and produce accurate predictions.

## Q: Why do we apply fit_transform() on training data and only transform() on test data?

Answer:
We calculate the mean and standard deviation only from the training data using fit(). Then we apply the same values to both training and test data using transform(). This prevents data leakage and ensures fair model evaluation.

## Flow of Data Preprocessing

Raw Data → Missing Values → Encoding → Train-Test Split → Feature Scaling → Train Model → Prediction ✅
