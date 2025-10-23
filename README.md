

# Predicting Salary Project

## 📘 Overview

This project focuses on **predicting job salaries** using **neural networks** that combine **categorical** and **textual (NLP)** features.
The dataset used comes from Kaggle:
👉 [Job Salary Prediction Dataset](https://www.kaggle.com/c/job-salary-prediction/data)

The goal is to build and compare multiple models that process structured and unstructured data to achieve accurate salary predictions.

---

## 🧠 Project Objectives

1. Conduct a short **Exploratory Data Analysis (EDA)** to understand dataset structure and key relationships.
2. Build **baseline and advanced neural network models** using categorical and text-based features.
3. Experiment with **different text representation techniques** (TF-IDF, embeddings, pretrained models).
4. Optimize performance using **custom training loops**, **early stopping**, and **hyperparameter tuning**.
5. Focus mainly on **RMSE** as the key evaluation metric, while keeping flexibility to compute others.

---

## 📊 1. Exploratory Data Analysis (EDA)

* Examined each column’s meaning, value ranges, and distributions.
* Conducted univariate and bivariate visualizations using `matplotlib` and `seaborn`.
* Checked for missing values and outliers.
* Basic text preprocessing overview for description and title columns.

---

## 🧩 2. Modeling Approach

### **Part 1: Baseline Model (Categorical Features Only)**

* Built preprocessing pipelines for categorical data (encoding, scaling, imputing).
* Created a **custom PyTorch dataset** and **feedforward neural network**.
* Implemented **custom early stopping** and **RMSE-based evaluation**.
* Experimented with hyperparameters such as learning rate, hidden layer sizes, batch size, and dropout.

---

### **Part 2: Text Integration**

Text columns added:

* **Full Description**
* **Job Title**

#### 🔹 Text Representation Techniques

1. **TF-IDF Vectorization** + optional dimensionality reduction using **Truncated SVD**.
2. **Sentence Embeddings** via **Sentence Transformers**.

* Created pipelines for text preprocessing and vectorization.
* Built models with **multi-input architectures** combining categorical and text data.
* Compared models using **scaled** and **unscaled** target values.

#### 🔹 Self-Taught & Pretrained Embeddings

* Trained a **custom Word2Vec model** on the dataset to learn domain-specific embeddings.

* Used **pretrained FastText embeddings** to capture semantic relationships.

* Manually **tokenized text data**, built a **vocabulary**, and **included the embedding matrix directly in the neural network**.

* Explored three embedding integration strategies:

  1. Using **precomputed embeddings** as direct model input.
  2. Using a **frozen embedding layer** initialized with pretrained weights.
  3. Using a **trainable embedding layer** for fine-tuning.

* Tested multiple architectures:

  * Multi-input simple networks
  * CNN-based networks
  * Networks with residual connections
  * Combinations of the above

* Applied regularization, dropout, and normalization to reduce overfitting.

* Visualized and compared model performance curves.

---

## 🧪 Training & Evaluation

* Implemented a **custom training loop** using PyTorch.
* Main metric: **Root Mean Squared Error (RMSE)**.
* Additional metrics available: MAE
* Used **ReduceLROnPlateau** scheduler for dynamic learning rate adjustment.
* Logged training progress and visualized loss and validation curves.

---

## ⚙️ Technologies Used

### **Programming & Frameworks**

* **Python 3.12** (recommended ≤ 3.12 for gensim compatibility)
* **PyTorch** – neural network modeling and training
* **scikit-learn** – preprocessing, pipelines, and evaluation
* **gensim** – Word2Vec and FastText embeddings
* **sentence-transformers** – sentence-level embeddings
* **category_encoders** – target encoding for categorical variables
* **joblib** – saving and loading models

### **Data & NLP Tools**

* **pandas**, **NumPy** – data handling and transformations
* **NLTK** – tokenization, lemmatization, stopword removal
* **TF-IDF**, **TruncatedSVD** – feature extraction and dimensionality reduction

### **Visualization & Analysis**

* **matplotlib**, **seaborn** – EDA and training visualization

---

## 📂 Repository Structure

```
PROJECT2_NEURAL_NETWORK_PREDICTION/
├── config/                               # Configuration files
│   └── config.py                         # Global constants and paths
│
├── data/                                 # Raw dataset
│   └── Train_rev1.csv                    # Main training dataset
│
├── embeddings/                           # Pre-trained or learned word embeddings
│
├── preprocessors/                        # Saved preprocessing objects 
│
├── tfidf_features/                       # Stored TF-IDF features
│
├── models/                               # Saved trained models 
│
├── notebooks/                            # Jupyter notebooks
│   └── nlp_salary_prediction_project.ipynb  # Main analysis and modeling notebook
│
├── reports/                              # Reports and visual results
│   ├── figures/                          # Generated plots and charts
│   ├── all_models_results.csv             # Summary of all model evaluations
│   ├── project_report.md                 # Detailed project report
│   └── model_naming.md                   # Documentation for model naming conventions
│
├── src/                                  # Core source code
│   ├── data_preprocessing/               # Data loading and preprocessing scripts
│   │   ├── loader.py                     # Loads raw data
│   │   ├── preprocess_categorical.py     # Encodes categorical features
│   │   ├── preprocess_target.py          # Preprocesses target variable (salary)
│   │   ├── splitter.py                   # Splits dataset into train/validation/test
│   │   └── text_features.py              # Extracts textual features
│   │
│   ├── datasets/                         # Custom dataset wrappers for PyTorch 
│   │   ├── multi_input_dataset.py
│   │   ├── pre_encoded_dataset.py
│   │   ├── salary_dataset.py
│   │   └── tokens_dataset.py
│   │
│   ├── models/                           # Neural network architectures
│   │   ├── embedding_matrix_model.py
│   │   ├── integrated_model.py
│   │   ├── multi_input_model.py
│   │   ├── pre_encoded_model.py
│   │   ├── self_taught_models.py
│   │   ├── simple_regressors.py
│   │   ├── unfrozen_models.py
│   │   └── residual_block.py
│   │
│   ├── pipeline/                         # Feature pipeline assembly
│   │   └── feature_pipeline.py
│   │
│   ├── preprocessors/                    # NLP preprocessing utilities
│   │   ├── pretrained_utils.py
│   │   ├── text_embedder.py
│   │   ├── text_preprocessors.py
│   │   ├── tfidf_transformer.py
│   │   ├── vocabulary.py
│   │   └── word2vec_utils.py
│   │
│   ├── training/                         # Training and evaluation logic
│   │   ├── early_stopping.py
│   │   ├── evaluate_model.py
│   │   └── train_model.py
│   │
│   └── utils/                            # Helper functions
│       ├── device_utils.py               # Device (MPS/CPU/GPU) handling
│       ├── logging_utils.py              # Logging configuration
│       ├── metrics_utils.py              # Custom evaluation metrics
│       ├── plot_utils.py                 # Plotting utilities
│       └── seed_utils.py                 # Reproducibility (random seeds)
│   
│
├── requirements.txt                      # Project dependencies
├── orchestrator.py                       # Main pipeline orchestrator (runs full workflow)
├── README.md                             # Project overview
└── .gitignore                            # Ignored files and directories for Git

```
---

## 📈 Results

The detailed **analysis of results** and performance comparison of all models are presented in
**`reports/project_report.md`**.

---


