# Alphabet Soup Charity - Predictive Analysis

## Overview
This project involves developing a binary classification model to predict whether an organization funded by Alphabet Soup will be successful based on various features in the dataset. The project follows a series of steps for data preprocessing, model compilation, training, evaluation, and optimization to achieve a target predictive accuracy higher than 75%.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Compilation, Training, and Evaluation](#model-compilation-training-and-evaluation)
- [Optimization](#optimization)
- [Results](#results)
- [Summary](#summary)
- [References](#references)

## Dependencies
To run this project, you need the following Python packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `h5py`

You can install these dependencies using the following command:
```bash
pip install pandas numpy scikit-learn tensorflow h5py
```

## Data Preprocessing
1. **Read Data:** Import and read the `charity_data.csv` file into a Pandas DataFrame.
2. **Drop Columns:** Remove non-beneficial columns, 'EIN' and 'NAME'.
3. **Unique Values:** Determine the number of unique values in each column.
4. **Combine Rare Categorical Variables:**
   - For `APPLICATION_TYPE`, replace rare occurrences with "Other".
   - For `CLASSIFICATION`, replace rare occurrences with "Other".
5. **Encode Categorical Variables:** Use `pd.get_dummies()` to convert categorical data to numeric.
6. **Split Data:** Divide the preprocessed data into features (`X`) and target (`y`) arrays. Split these arrays into training and testing datasets.
7. **Scale Data:** Use `StandardScaler` to scale the features.

## Model Compilation, Training, and Evaluation
1. **Define Model:** Create a neural network model with input features and hidden layers.
2. **Hidden Layers:** Add hidden layers with appropriate activation functions.
3. **Output Layer:** Add an output layer with an appropriate activation function.
4. **Compile Model:** Compile the model with a suitable optimizer and loss function.
5. **Train Model:** Train the model using the training data.
6. **Evaluate Model:** Evaluate the model using the test data to determine loss and accuracy.
7. **Save Model:** Save the trained model to an HDF5 file named `AlphabetSoupCharity.h5`.

## Optimization
1. **Repeat Preprocessing:** Repeat the data preprocessing steps.
2. **Optimize Model:** Implement at least three optimization methods, such as:
   - Adjusting the number of neurons.
   - Adding more hidden layers.
   - Changing activation functions.
   - Modifying the number of epochs.
3. **Save Optimized Model:** Save the optimized model to an HDF5 file named `AlphabetSoupCharity_Optimisation.h5`.

## Results
### Data Preprocessing
- **Target Variable:** The target variable is `IS_SUCCESSFUL`.
- **Feature Variables:** The feature variables include all other columns after preprocessing.
- **Dropped Variables:** The `EIN` and `NAME` columns were dropped.

### Model Compilation, Training, and Evaluation
- **Model Structure:** The model includes multiple hidden layers with ReLU activation functions and an output layer with a sigmoid activation function.
- **Model Performance:** The initial model performance did not meet the 75% accuracy target.
- **Optimization Attempts:** Various methods were employed to optimize the model, such as increasing neurons, adding layers, and changing activation functions.

## Summary
The deep learning model was trained to predict the success of organizations funded by Alphabet Soup. Despite various optimization attempts, the model's accuracy was evaluated and reported. Further improvements could be made by experimenting with different models or feature engineering techniques.

## References
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [University of Adelaide Git bootcamp](https://git.bootcampcontent.com/University-of-Adelaide/UADEL-VIRT-DATA-PT-12-2023-U-LOLC/-/tree/main/20-Supervised-Learning?ref_type=heads)
