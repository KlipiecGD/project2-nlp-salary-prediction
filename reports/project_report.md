# Project Report: Job Salary Prediction with Neural Networks 

## 1. Introduction and Objectives
This report details the development and evaluation of neural network models designed to predict job salaries based on a combination of categorical and textual features. The dataset utilized originates from Kaggle's Job Salary Prediction competition. 
The main objectives of this project were to:
- Perform exploratory data analysis (EDA) to understand the dataset.
- Develop baseline and advanced neural network models incorporating both categorical and text-based features.
- Experiment with various text representation techniques, including TF-IDF and sentence embeddings.
- Optimize model performance through custom training loops, early stopping, and hyperparameter tuning.

## 2. Data Overview
The dataset comprises various features including job titles, full job descriptions, and categorical attributes such as location, contract type, and job category. Initial exploratory data analysis (EDA) was conducted to understand the data distribution, identify missing values, and visualize key relationships.

## 3. Modeling Approach
### 3.1 Baseline Model (Categorical Features Only)
A baseline feedforward neural network was constructed using only the categorical features. Preprocessing steps included filling missing values, encoding categorical variables, and scaling numerical features. A custom PyTorch dataset class was implemented to facilitate data loading. 

### 3.2 Text Integration
To enhance the model, textual features from job titles and full descriptions were incorporated. Some approaches were tried:
- **TF-IDF Vectorization**: Text data was vectorized using TF-IDF
- **Sentence Embeddings from Hugging Face**: Leveraged pre-trained Sentence Transformers to obtain dense vector representations of the text.
- **Self-Taught Models**: Explored architectures that learn text representations jointly with the main task.
- **Word2Vec**: Trained a custom Word2Vec model on the dataset to capture domain-specific semantics.
- **Readily Available Pretrained Models**: Utilized models like fastText for text representation.

### 3.3 Model Architectures
Multiple architectures were experimented with, including:
- Simple feedforward networks for categorical data.
- Multi-input models combining categorical and text features.
- Residual blocks to improve learning in deeper networks.
- Convolutional layers for text feature extraction.

## 4. Training and Evaluation
Models were trained using custom training loops with early stopping based on validation RMSE. Hyperparameter tuning was performed to optimize learning rates, batch sizes, and network architectures. The primary evaluation metric was MSE (RMSE for readability in original units).

## 5. Results overview

### 5.1 Categorical Models Performance
Baseline models using only categorical features achieved an RMSE of X on the validation set, serving as a benchmark for subsequent models.

**Model information:**
- Architecture: Simple Regressor - Feedforward Neural Network with 2 Hidden Layers, dropout and ReLU activations
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 32, Hidden Layers = [128, 64], Dropout = 0.2, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 11 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_baseline_cat.png)

#### 5.1.1 Hyperparameter Tuning Results
The hyperparameter tuning process revealed that for categorical data, changes in hyperparameters do not significantly affect model performance. Tried configurations included (hyperparameter that were not mentioned were kept as in the baseline model, expect for batch size which after training model number 3 was set to 64 to speed up training for the experiments):
- Not hyperparameter tuning, other approach for filling missing values - 'most frequent' value - Model2
- Larger batch size - 64 - Model3
- Less neurons in hidden layers - [64, 32] - Model4
- SGD optimizer - Model5
- Enabling learning rate scheduler - Model6

The loss curves for these experiments are shown below:

- **Most Frequent Value Imputation**: ![alt text](figures/curves_cat_mostfrequent.png)
- **Larger Batch Size**: ![alt text](figures/curves_cat_batchsize64.png)
- **Fewer Neurons**: ![alt text](figures/curves_cat_fewer_neurons.png)
- **SGD Optimizer**: ![alt text](figures/curves_cat_sgd.png)
- **Learning Rate Scheduler**: ![alt text](figures/curves_cat_lr_scheduler.png)

Below you can see plots that compare Train, Validation and Test RMSE for all categorical models, their training times, number of parameters and overfitting levels.

- **RMSE Comparison:** ![alt text](figures/all_cat_models_rmse.png)
- **Training Time Comparison:** ![alt text](figures/all_cat_models_time.png)
- **Number of Parameters Comparison:** ![alt text](figures/all_cat_models_params.png)
- **Overfitting Level Comparison:** ![alt text](figures/all_cat_models_overfitting.png)



### 5.2 TF-IDF and Sentence Transformers Models Performance

### 5.2.1 TF-IDF Models

### 5.2.2 Sentence Transformers Models

### 5.2.3 Analysis of TF-IDF and Sentence Transformers Models Performance

### 5.3 Self-Taught, Word2Vec, and Pretrained Embeddings Models Performance

#### 5.3.1 Self-Taught Models

#### 5.3.2 Word2Vec Embeddings Models

#### 5.3.3 Pretrained Embeddings Models

#### 5.3.4 Analysis of Self-Taught, Word2Vec, and Pretrained Embeddings Models Performance

### 5.4 Mitigation of Overfitting

### 5.5 Comparative Analysis

#### 5.5.1 Best 10 Models Summary
The plots below summarize performance, training time, number of parameters and overfitting levels for the best 10 models developed during this project.
- **RMSE Comparison:** ![alt text](figures/best10_models_rmse.png)
- **Training Time Comparison:** ![alt text](figures/best10_models_time.png)
- **Number of Parameters Comparison:** ![alt text](figures/best10_models_params.png)
- **Overfitting Level Comparison:** ![alt text](figures/best10_models_overfitting.png)

## 6. Conclusions

