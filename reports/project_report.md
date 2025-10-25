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
- Simple feedforward single input networks - `SimpleRegressor`
- Feedforward networks with Batch Normalization - `SimpleRegressorWithNormalization`
- Deeper feedforward networks with additional hidden layers and batch normalization - `IntegratedNN`
- Multi-input models combining categorical and text features -`MultiInputNN`
- Residual blocks to improve learning in deeper networks - `ResidualBlock`
- Model for self-taught learned embeddings during training - `SelfTaughtNN`
- Model with ready made frozen/unfrozen embedding matrix - `PreEncodedModel`
- Self-taught and unfrozen embedding matrix improved - residual connections/cnn/cnn+residual conncetions

## 4. Training and Evaluation
Models were trained using custom training loops with early stopping based on validation RMSE. Hyperparameter tuning was performed to optimize learning rates, batch sizes, and network architectures. The primary evaluation metric was MSE (RMSE for readability in original units).

## 5. Results overview

### 5.1 Categorical Models Performance
Baseline models using only categorical features achieved an RMSE of X on the validation set, serving as a benchmark for subsequent models.

**Baseline Categorical Model information - Model 1:**
- Architecture: Simple Regressor 
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

As mentioned, hyperparameter tuning did not lead to significant improvements in performance for categorical models. The baseline model remained the best performer among the categorical-only models.

### 5.2 TF-IDF and Sentence Transformers Models Performance

To incorporate text data, at the start two strategies were deployed: TF-IDF and Sentence Transformers.

### 5.2.1 TF-IDF Models

I created class that transforms FullDescripition and Title cols using tf-idf with max 800 features for description and 50 for title. Text was cleaned and preprocessed (e.g., lowercasing, removing punctuation) by tf-idf and also with custom function. The resulting sparse matrices were then converted to dense format and combined with categorical features for model training. Then baseline model was trained

**Baseline TF-IDF Model information - Model 1:**
- Architecture: Simple Regressor 
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Hidden Layers = [128, 64], Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 11 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_baseline_tfidf.png)

After training with simple architecture I trained it by model with additional batch normalization to prevent overfitting and improve performance

**TF-IDF with Batch Normalization Model information - Model 2:**
- Architecture: Simple Regressor with Batch Normalization
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Hidden Layers = [128, 64], Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 11 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_tfidf_batchnorm.png)

The next tried Tf-idf model was deeper network with additional hidden layer and batch normalization

**Deeper TF-IDF Model with Batch Normalization - Model 3:**
- Architecture: IntegratedNN
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Hidden Layers = [256, 128, 64], Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_tfidf_integrated.png)

I also tried to pass tf-idf text features through separate input branch in multi-input model.

**TF-IDF Multi-Input Model - Model 4:**
- Architecture: MultiInputNN
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Embedding Hidden Layers = [256, 128], Tabular Hidden = [64, 32],
        Combined Hidden = [128, 64], dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_tfidf_multiinput.png)

After these experiments I peformed hyperparameter tuning and other techniques for tf-idf using IntegratedNN architecture. Below you can find tried approaches and loss curves for each of them

- filling missing values with most frequent value instead of 'unknown' - Model 5
![alt text](figures/curves_tfidf_mostfrequent.png)
- Simpler model with smaller hidden size - SimpleRegressorWithNormalization with [64, 32] neurons - Model 6
![alt text](figures/curves_tfidf_simple.png)
- IntegratedNN with less neurons - Model 7
![alt text](figures/curves_tfidf_integrated_small.png)
- Using learning rate scheduler - Model 8
![alt text](figures/curves_tfidf_scheduler.png)
- Using SGD optimizer - Model 9
![alt text](figures/curves_tfidf_sgd.png)
- Adding SVD for dimensionality reduction - Model 10
![alt text](figures/curves_tfidf_svd.png)
- Removing stop words in text features - Model 11
![alt text](figures/curves_tfidf_stopwords.png)

As an additional experiment I also trained tf-idf models for preciting log of the salary instead scaling target using approach and arcitecture from Model 12 - Log Model.
The loss curves for these experiments are shown below:
![alt text](figures/curves_tfidf_log.png)

#### Comparison of TF-IDF Models

The plots below summarize performance, training time, number of parameters and overfitting levels for all TF-IDF models developed during this project.
- **RMSE Comparison:** ![alt text](figures/all_tfidf_models_rmse.png)
- **Training Time Comparison:** ![alt text](figures/all_tfidf_models_time.png)
- **Number of Parameters Comparison:** ![alt text](figures/all_tfidf_models_params.png)
- **Overfitting Level Comparison:** ![alt text](figures/all_tfidf_models_overfitting.png)

We can see that the best performing TF-IDF model was Model 11 - Integrated NN with stop words removing. We can also notice that Model 4 and Model 9 may need longer training as validation loss is smaller than training loss.

### 5.2.2 Sentence Transformers Models

For sentence transformers models I used pretrained model from Hugging Face - all-MiniLM-L12-v2 to generate sentence embeddings for FullDescription and Title columns. I tried to pass these embeddings through separate input branch in multi-input model and also concatenate them with categorical features as single input.

As a baseline I trained SimpleRegressor model with concatenated categorical and sentence embeddings features.

**Baseline Sentence Transformers Model information - Model 1:**
- Architecture: Simple Regressor 
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Hidden Layers = [128, 64], Dropout = 0.2, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 11 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_baseline_sentence_transformers.png)

Next I trained model with batch normalization - SimpleRegressorWithNormalization

**Sentence Transformers with Batch Normalization Model information - Model 2:**
- Architecture: Simple Regressor with Batch Normalization
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Hidden Layers = [128, 64], Dropout = 0.2, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 11 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_sentence_transformers_batchnorm.png)

Then I trained IntegratedNN model with sentence embeddings and categorical features as single input.

**Deeper Sentence Transformers Model with Batch Normalization - Model 3:**
- Architecture: IntegratedNN
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Hidden Layers = [256, 128, 64], Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 14 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_sentence_transformers_integrated.png)

As an experiment I tried to put batch normalization after Relu activation in IntegratedNN architecture instead of before it.

**Sentence Transformers IntegratedNN with BatchNorm after Relu - Model 4:**
- Architecture: IntegratedNN with BatchNorm after Relu
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Hidden Layers = [256, 128, 64], Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 13 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_sentence_transformers_integrated_batchnorm_after_relu.png)

Finally I trained MultiInputNN model with separate input branch for sentence embeddings and categorical features.

**Sentence Transformers Multi-Input Model - Model 5:**
- Architecture: MultiInputNN
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Embedding Hidden Layers = [256, 128], Tabular Hidden = [64, 32],
        Combined Hidden = [128, 64], dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_sentence_transformers_multiinput.png)

#### Comparison of Sentence Transformers Models

The plots below summarize performance, training time, number of parameters and overfitting levels for all Sentence Transformers models developed during this project.
- **RMSE Comparison:** ![alt text](figures/all_sentence_transformers_models_rmse.png)
- **Training Time Comparison:** ![alt text](figures/all_sentence_transformers_models_time.png)
- **Number of Parameters Comparison:** ![alt text](figures/all_sentence_transformers_models_params.png)
- **Overfitting Level Comparison:** ![alt text](figures/all_sentence_transformers_models_overfitting.png)

The best model among sentence transformers was Model 3 - IntegratedNN with classic batch normalization before Relu activation.
The model with inverted batch normalization (Model 4) showed that after some epochs valid loss becomes much larger than training loss which may indicate that this approach is not effective in this case.

### 5.3 Self-Taught, Word2Vec, and Pretrained Embeddings Models Performance

#### 5.3.1 Self-Taught Models

The next approach was to tokenize text data using simple manual splitting tokens and allow model to learn embeddings from scratch during training. I created custom functions  that tokenizes text data, builds vocabulary. Then I trained SelfTaughtNN which has a `nn.Embedding` layer that learns embeddings during training.

**Baseline Self-Taught Embeddings Model information - Model 1:**
- Architecture: SelfTaughtNN
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Embedding Size = 256, min_freq=20 (minimum frequency for token inclusion) Regressor Hidden Layers = [256, 128], Categorical Hidden Layer = 128 Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3, Scheduler = Yes, Scheduler patience = 1, factor=0.5
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_selftaught_baseline.png)

#### 5.3.2 Word2Vec Embeddings Models

For Word2Vec I experimented with 3 different approaches:
1. Training Word2Vec model (skip-gram for each approach) on our dataset to transform text data into embeddings and pass them thorough PreEncodedModel architecture which doesn't have embedding layer.

2. Training Word2Vec model on our dataset to build embedding matrix for our vocabulary and use it in the model as a frozen embedding layer.

3. Training Word2Vec model on our dataset to build embedding matrix for our vocabulary and use it in the model as a trainable embedding layer.

**Word2Vec with embeddings as input information - Model 1:**
- Architecture: PreEncodedModel
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, min_freq=20 (minimum frequency for token inclusion), Embedding Size = 256, Regressor Hidden Layers = [256, 128], Categorical Hidden Layer = 128 Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3, Scheduler = Yes, Scheduler patience = 1, factor=0.5
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_word2vec_embeddings_input.png)

**Word2Vec with frozen embedding layer information - Model 2:**
- Architecture: EmbeddingMatrixNN with frozen embeddings
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, min_freq=20 (minimum frequency for token inclusion), Embedding Size = 256, Regressor Hidden Layers = [256, 128], Categorical Hidden Layer = 128 Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3, Scheduler = Yes, Scheduler patience = 1, factor=0.5
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_word2vec_frozen_embedding.png) 

**Word2Vec with trainable embedding layer information - Model 3:**

- Architecture: EmbeddingMatrixNN with trainable embeddings
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, min_freq=20 (minimum frequency for token inclusion), Embedding Size = 256, Regressor Hidden Layers = [256, 128], Categorical Hidden Layer = 128 Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3, Scheduler = Yes, Scheduler patience = 1, factor=0.5
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_word2vec_trainable_embedding.png)

From the plots we can see that the model with unfrozen embedding layer performed the best among Word2Vec approaches, but is most prone to overfitting.

#### 5.3.3 Pretrained Embeddings Models

For pretrained embeddings I used fastText model - 'fasttext-wiki-news-subwords-300' to build embedding matrix for our vocabulary and use it in the same way as Word2Vec approaches - transforming tokens into embeddings and using them in both frozen and trainable embedding layers.

**Pretrained Embeddings as input information - Model 1:**
- Architecture: PreEncodedModel
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Embedding Size = 300, Regressor Hidden Layers = [256, 128], Categorical Hidden Layer = 128 Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3, Scheduler = Yes, Scheduler patience = 1, factor=0.5
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_pretrained_embeddings_input.png)

**Pretrained Embeddings with frozen embedding layer information - Model 2:**

- Architecture: EmbeddingMatrixNN with frozen embeddings
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Embedding Size = 300, Regressor Hidden Layers = [256, 128], Categorical Hidden Layer = 128 Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3, Scheduler = Yes, Scheduler patience = 1, factor=0.5
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_pretrained_frozen_embedding.png)

**Pretrained Embeddings with trainable embedding layer information - Model 3:**

- Architecture: EmbeddingMatrixNN with trainable embeddings
- Missing Value Handling: Imputation with 'unknown' for categorical features
- Hyperparameters: Learning Rate = 0.001, Batch Size = 64, Embedding Size = 300, Regressor Hidden Layers = [256, 128], Categorical Hidden Layer = 128 Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 3, Scheduler = Yes, Scheduler patience = 1, factor=0.5
- Number of Trainable Parameters: X
- Training Time: X minutes (stopped by early stopping after 12 epochs)
- Training RMSE: X
- Validation RMSE: X    
- Test RMSE: X
- Loss Curve: ![alt text](figures/curves_pretrained_trainable_embedding.png)

Again model with trainable embedding layer performed the best among Pretrained Embeddings approaches, but is most prone to overfitting. However it didn't outperform Word2Vec with trainable embeddings.

### 5.3.4 Hyperparameter Tuning for Self-Taught model

To experiment with hyperparameter tuning for Self-Taught model I tried the following approaches and architectures:

- initial cleaning - removing stop words - SelfTaughtModelv2
    - ![alt text](figures/curves_selftaught_stop_words.png)

- higher embedding size - 300 - SelfTaughtModelv3
    - ![alt text](figures/curves_selftaught_embedding300.png)

- smaller embedding size - 128 - SelfTaughtModelv4
    - ![alt text](figures/curves_selftaught_embedding128.png)

- larger vocabulary - min_freq=15 - SelfTaughtModelv5
    - ![alt text](figures/curves_selftaught_minfreq15.png)

- shorter max sequence length - 250 - SelfTaughtModelv6
    - ![alt text](figures/curves_selftaught_maxlen250.png)

From the plots we can see that increasing embedding size to 300 improved performance. Other approaches did not lead to significant improvements.

#### 5.3.4 Mitigation of Overfitting for Self-Taught models with Trainable Embeddings

All the models that learn embeddings during training or modify already learned embeddings showed signs of overfitting. To mitigate this, I experimented with SelfTaughtNN architecture by:

- increasing dropout rate from 0.3 to 0.45 - SelfTaughtModelOverfittingMitigationv1
    - ![alt text](figures/curves_selftaught_dropout45.png)

- decreasing number of neurons in hidden layers: categorical hidden from 128 to 64 and regressor hidden from [256, 128] to [128, 64] - SelfTaughtModelOverfittingMitigationv2
    - ![alt text](figures/curves_selftaught_fewer_neurons.png)

- Starting with larger learning rate = 0.01 and using learning rate scheduler to reduce it by factor 0.1 each time (patience=0) - SelfTaughtModelOverfittingMitigationv3
    - ![alt text](figures/curves_selftaught_lr_scheduler.png)

- increasing vocabulary size by decreasing min_freq from 20 to 10 - SelfTaughtModelOverfittingMitigationv4
    - ![alt text](figures/curves_selftaught_minfreq10.png)

- early stopping with patience = 2, dropout = 0.4, min_freq = 10, lr_scheduler_patience = 0, scheduler_factor = 0.25, lr = 0.005,  - SelfTaughtModelOverfittingMitigationv5
    - ![alt text](figures/curves_selftaught_combined.png)

- early stopping with patience = 2, dropout = 0.35, min_freq = 20, lr_scheduler_patience = 0, scheduler_factor = 0.5, lr = 0.003, - SelfTaughtModelOverfittingMitigationv6
    - ![alt text](figures/curves_selftaught_combined_v2.png)

This investigation shows that it is very challenging to mitigate overfitting and keep the validation RMSE low.


#### 5.3.5 Comparison of Self-Taught, Word2Vec and Pretrained Embeddings Models

The plots below summarize performance, training time, number of parameters and overfitting levels for all Self-Taught, Word2Vec and Pretrained Embeddings including SelfTaught hyperparameter tuning and overfitting mitigation models.

- **RMSE Comparison:** ![alt text](figures/all_selftaught_word2vec_pretrained_models_rmse.png)
- **Training Time Comparison:** ![alt text](figures/all_selftaught_word2vec_pretrained_models_time.png)
- **Number of Parameters Comparison:** ![alt text](figures/all_selftaught_word2vec_pretrained_models_params.png)
- **Overfitting Level Comparison:** ![alt text](figures/all_selftaught_word2vec_pretrained_models_overfitting.png)

### 5.5 New more Complex Architectures for Self-Taught, Word2Vec and Pretrained Embeddings

To further improve performance, I experimented with more complex architectures for the best performing approaches from previous section - SelfTaughtNN with trainable embeddings, Word2Vec with trainable embeddings and Pretrained Embeddings with trainable embeddings. I added residual connections, convolutional layers and combination of both to these models. For each of these models I set the same hyperparameters:
- Learning Rate = 0.001, Batch Size = 64, Embedding Size = 300, Regressor Hidden Layers = [256, 128], Categorical Hidden Layer = 128 Dropout = 0.3, Optimizer = Adam, Loss Function = MSE, Early Stopping Patience = 2, Scheduler = Yes, Scheduler patience = 0, factor=0.5

The following architectures were tried:

- SelfTaughtNN with Residual Connections - ResidualSelfTaughtModelv1
    - ![alt text](figures/curves_selftaught_residual.png)

- SelfTaughtNN with Convolutional Layers - CNNSelfTaughtModelv1
    - ![alt text](figures/curves_selftaught_cnn.png)

- SelfTaughtNN with Convolutional Layers and Residual Connections - CNNResidualSelfTaughtModelv1
    - ![alt text](figures/curves_selftaught_cnn_residual.png)

- Word2Vec with Residual Connections - ResidualWord2VecModelv1
    - ![alt text](figures/curves_word2vec_residual.png)

- Word2Vec with Convolutional Layers - CNNWord2VecModelv1
    - ![alt text](figures/curves_word2vec_cnn.png)

- Word2Vec with Convolutional Layers and Residual Connections - CNNResidualWord2VecModelv1
    - ![alt text](figures/curves_word2vec_cnn_residual.png)

- FastText Pretrained Embeddings with Residual Connections - ResidualFastTextModelv1
    - ![alt text](figures/curves_pretrained_residual.png)

- FastText Pretrained Embeddings with Convolutional Layers - CNNFastTextModelv1
    - ![alt text](figures/curves_pretrained_cnn.png)

- FastText Pretrained Embeddings with Convolutional Layers and Residual Connections - CNNResidualFastTextModelv1
    - ![alt text](figures/curves_pretrained_cnn_residual.png)


**Comparison of New Complex Architectures**

The plots below summarize performance, training time, number of parameters and overfitting levels for new complex architectures.
- **RMSE Comparison:** ![alt text](figures/complex_architectures_models_rmse.png)
- **Training Time Comparison:** ![alt text](figures/complex_architectures_models_time.png)
- **Number of Parameters Comparison:** ![alt text](figures/complex_architectures_models_params.png)
- **Overfitting Level Comparison:** ![alt text](figures/complex_architectures_models_overfitting.png)

We can see that these models are more complex and have larger number of parameters which leads to longer training times. However some of them achieved better performance compared to simpler architectures. The best performing model among these was CNNResidualSelfTaughtModelv1.

### 5.6 Comparative Analysis

The plots below summarize performance, training time, number of parameters and overfitting levels for the best 10 models developed during this project.
- **RMSE Comparison:** ![alt text](figures/best10_models_rmse.png)
- **Training Time Comparison:** ![alt text](figures/best10_models_time.png)
- **Number of Parameters Comparison:** ![alt text](figures/best10_models_params.png)
- **Overfitting Level Comparison:** ![alt text](figures/best10_models_overfitting.png)

## 6. Conclusions

* The best approach for text representation was using Word2vec, training it on our dataset, building an embedding matrix for our vocabulary and using it in the model as a trainable embedding layer. This approach allowed the model to learn domain-specific semantics, leading to improved performance compared to using generic pretrained embeddings or simpler methods like TF-IDF.

* Simple manual splitting tokens and allowing model to learn embeddings from scratch also performed really well. It outperformed sophisticated pretrained models from Hugging Face like Sentence Transformers. This suggests that for this specific task, domain-specific learned embeddings were more effective than general-purpose pretrained models.

* The model that achieved the best performance was X, with a Test RMSE of Y. However, it exhibited signs of overfitting, as indicated by the gap between training and validation RMSE which was Z. Moreover, the training time for this model was A minutes, which is quite long. Least but not least, the model had B trainable parameters, making it relatively complex.

* If we prioritize model simplicity and training efficiency, Model C and Model D stand out. Model C had a Test RMSE of E, with a training time of F minutes and G trainable parameters. Model D achieved a Test RMSE of H, with a training time of I minutes and J trainable parameters. Both models demonstrated a good balance between performance and complexity, making them suitable choices for deployment in resource-constrained environments.

* Overall, the experiments highlighted the importance of tailored text representation techniques and careful model architecture selection in achieving optimal performance for job salary prediction tasks. For only categorical features the baseline achieved the test RMSE of K which shows while as mentioned best models with text features achieved X test RMSE. It is significant improvement showing the value of incorporating textual data in the prediction task.


