# Student-Grade-Prediction

## Using Machine Learning for Grade Prediction with SVR

### Setting Up the Environment

- Import essential libraries:
  - `pandas` for data manipulation  
  - `seaborn` and `matplotlib` for data visualization  
  - `scikit-learn` for machine learning  
  - `pickle` for model saving/loading  
- Create custom classes:
  - `Model`: for preprocessing, training, and prediction  
  - `Pickle`: inherits from `Model` and adds model persistence  

### Reading and Cleaning Data

- Create an object of the `Model` class using `"data.csv"`  
- Rename dataset columns using `rename_columns()`  
- Save the cleaned DataFrame to `"data2.csv"`  

### Selecting Features and Target

- Features used:
  - `Stu_Hours`
  - `Extra_Hours`
- Target variable:
  - `Grades`

### Splitting and Training the Model

- Split the data into training/testing sets using `split_data()`  
- Train the model using `train_svm_model()`  

### Making Predictions

- Use the `predict()` method to predict on test set  

### Saving and Loading the Model with Pickle

- Create an object of the `Pickle` class  
- Rename columns again, split data, and train  
- Save the trained model using `store_model("svr_model.pkl")`  
- Load the saved model using `load_model("svr_model.pkl")`  
- Predict using the loaded model with `predict_with_loaded_model(X_test)`  

### Final Output

- Display predictions with:
  -python
  print("Predictions with loaded model:", predictions)
