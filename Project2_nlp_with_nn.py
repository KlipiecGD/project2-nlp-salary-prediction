#!/usr/bin/env python
# coding: utf-8

# # Project 2: NLP with Neural Networks

# ## Importing necessary libraries and loading the dataset

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import random
import re
import string
import time

from typing import Optional, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from category_encoders import TargetEncoder

from sentence_transformers import SentenceTransformer

import joblib

import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download once
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


RANDOM_SEED = 42
FILEPATH = "Train_rev1.csv"
MODELS_DIR = "models/"
PREPROCESSORS_DIR = "preprocessors/"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

torch.use_deterministic_algorithms(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[16]:


def set_seed(seed_value=RANDOM_SEED):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

set_seed()


# In[17]:


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(RANDOM_SEED)


# In[4]:


df = pd.read_csv(FILEPATH)  


# ## Exploratory Data Analysis (EDA)

# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.isnull().sum() / len(df) # Calculate the percentage of missing values for each column


# In[10]:


# Id column is not useful for prediction, so we will drop it
df = df.drop(columns=['Id'])


# In[11]:


# Checking for duplicates
df.duplicated().sum()


# In[12]:


df.nunique()


# ### Univariate Analysis

# #### Inspecting the 'Title' column

# In[13]:


df['Title'].value_counts()


# In[14]:


df[df['Title'].isna()]


# We can fill it manually.

# In[15]:


df.loc[df['Title'].isnull(), 'Title'] = 'Quality Improvement Manager'


# In[16]:


# Plot 10 most common titles
plt.figure(figsize=(10,6))
sns.countplot(y='Title', data=df, order=df['Title'].value_counts().index[:10])
plt.title('Top 10 Most Common Job Titles')
plt.xlabel('Count')
plt.ylabel('Job Title')
plt.show()


# #### Inspecting the 'FullDescription' column

# From the description on Kaggle, we now that *** hide salary information in description.

# In[17]:


df['FullDescription'].head()


# Check for some common words in IT job descriptions:

# In[18]:


words = ['python', 'java', 'c++', 'sql', 'html', 'css', 'tensorflow', 'pytorch']
for word in words:
    print(f"Number of job descriptions containing '{word}': {df['FullDescription'].str.contains(word, case=False).sum()}")


# #### Inspecting the 'LocationRaw' column

# In[19]:


df['LocationRaw'].head(10)


# In[20]:


df['LocationRaw'].value_counts()


# In[21]:


plt.figure(figsize=(10,6))
sns.countplot(y='LocationRaw', data=df, order=df['LocationRaw'].value_counts().index[:10])
plt.title('Top 10 Most Common Job Locations')
plt.xlabel('Count')
plt.ylabel('Location')
plt.show()


# #### Inspecting the 'LocationNormalized' column

# In[22]:


df['LocationNormalized'].head(10)


# In[23]:


df['LocationNormalized'].value_counts()


# In[24]:


plt.figure(figsize=(10,6))
sns.countplot(y='LocationNormalized', data=df, order=df['LocationNormalized'].value_counts().index[:10])
plt.title('Top 10 Most Common Job Locations')
plt.xlabel('Count')
plt.ylabel('Location')
plt.show()


# #### Inspecting the 'ContractType' column

# In[25]:


df['ContractType'].head(10)


# In[26]:


df['ContractType'].isnull().sum() / len(df) 


# In[27]:


df['ContractType'].value_counts()


# #### Inspecting the 'ContractTime' column

# In[28]:


df['ContractTime'].head(10)


# In[29]:


df['ContractTime'].isnull().sum() / len(df)


# In[30]:


df['ContractTime'].value_counts()


# #### Inspecting the 'Company' column

# In[31]:


df['Company'].head(10)


# In[32]:


df['Company'].isnull().sum() / len(df)


# In[33]:


df['Company'].value_counts()


# In[34]:


plt.figure(figsize=(10,6))
sns.countplot(y='Company', data=df, order=df['Company'].value_counts().index[:10])
plt.title('Top 10 Most Common Companies')
plt.xlabel('Count')
plt.ylabel('Company')
plt.show()


# #### Inspecting the 'Category' column

# In[35]:


df['Category'].head(10)


# In[36]:


df['Category'].value_counts()


# In[37]:


plt.figure(figsize=(10,6))
sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index[:10])
plt.title('Top 10 Most Common Categories')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()


# #### Inspecting the 'SalaryRow' column

# In[38]:


df['SalaryRaw'].head(10)


# In[39]:


df['SalaryRaw'].value_counts()


# #### Inspecting the 'SalaryNormalized' column

# In[40]:


df['SalaryNormalized'].head(10)


# In[41]:


df['SalaryNormalized'].describe()


# In[42]:


df[['SalaryRaw','SalaryNormalized']].head(10)


# In[43]:


df['SalaryNormalized'].value_counts()


# In[44]:


plt.figure(figsize=(10,6))
sns.histplot(df['SalaryNormalized'], bins=50, kde=True)
plt.title('Distribution of Normalized Salaries')
plt.xlabel('SalaryNormalized')
plt.ylabel('Frequency')
plt.show()


# In[45]:


plt.figure(figsize=(10,6))
sns.boxplot(y='SalaryNormalized', data=df)
plt.title('Boxplot of Normalized Salaries')
plt.ylabel('SalaryNormalized')
plt.show()


# I will also look how the salary is distributed after some transformations. Let's start with log transformation.

# ##### Log transformed salary distribution

# In[239]:


df['Salary_log1p'] = np.log1p(df['SalaryNormalized'])

plt.figure(figsize=(10,6))
sns.histplot(df['Salary_log1p'], bins=30, kde=True)
plt.title('Distribution of Log-Transformed Salary')
plt.xlabel('Log(Salary)')
plt.ylabel('Frequency')
plt.show()


# In[240]:


plt.figure(figsize=(10,6))
sns.boxplot(y=df['Salary_log1p'])
plt.title('Boxplot of Log-Transformed Salaries')
plt.ylabel('Log(Salary)')
plt.show()


# In[245]:


# to inverse if needed
original_salary = np.expm1(df['Salary_log1p'])
np.allclose(original_salary, df['SalaryNormalized'])


# ##### Standardized salary distribution

# We know that standardization and normalization don't change the shape of the distribution, so we will only plot it to check values.

# In[241]:


df['Salary_standardized'] = (df['SalaryNormalized'] - df['SalaryNormalized'].mean()) / df['SalaryNormalized'].std()

plt.figure(figsize=(10,6))
sns.histplot(df['Salary_standardized'], bins=30, kde=True)
plt.title('Distribution of Standardized Salary')
plt.xlabel('Standardized Salary')
plt.ylabel('Frequency')
plt.show()


# In[250]:


plt.figure(figsize=(10,6))
sns.boxplot(y=df['Salary_standardized'])
plt.title('Boxplot of Standardized Salaries')
plt.xlabel('Standardized Salary')
plt.show()


# ##### Normalized (Min-Max) salary distribution

# In[246]:


df['Salary_minmax'] = (df['SalaryNormalized'] - df['SalaryNormalized'].min()) / (df['SalaryNormalized'].max() - df['SalaryNormalized'].min())

plt.figure(figsize=(10,6))
sns.histplot(df['Salary_minmax'], bins=30, kde=True)
plt.title('Distribution of Min-Max Normalized Salary')
plt.xlabel('Min-Max Normalized Salary')
plt.ylabel('Frequency')
plt.show()


# In[247]:


plt.figure(figsize=(10,6))
sns.boxplot(y='Salary_minmax', data=df)
plt.title('Boxplot of Min-Max Normalized Salaries')
plt.ylabel('Min-Max Normalized Salary')
plt.show()


# #### Inspecting the 'SourceName' column

# In[46]:


df['SourceName'].head(10)


# In[47]:


df['SourceName'].isnull().sum() 


# In[48]:


df.loc[df['SourceName'].isnull()]


# In[49]:


df['SourceName'].value_counts()


# In[50]:


df['SourceName'].nunique()


# In[51]:


plt.figure(figsize=(10,6))
sns.countplot(y='SourceName', data=df, order=df['SourceName'].value_counts().index[:10])
plt.title('Top 10 Most Common Work Posting Websites')
plt.xlabel('Count')
plt.ylabel('SourceName')
plt.show()


# We have missing values only in categorical columns. As for now we will fill them with 'Unknown' value.

# In[52]:


cols_with_na = ["ContractType", "ContractTime", "Company", "SourceName"]

df[cols_with_na] = df[cols_with_na].fillna("Unknown")

df.isnull().sum()


# ### Bivariate Analysis

# In[53]:


df.info()


# In[54]:


# Avg Salary by Location
salaries_by_location = df.groupby('LocationNormalized')['SalaryNormalized'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=salaries_by_location.values, y=salaries_by_location.index)
plt.title('Top 10 Locations by Average Salary')
plt.xlabel('Average Salary')
plt.ylabel('Location')
plt.show()


# In[55]:


# Median Salary by Location
salaries_by_location_median = df.groupby('LocationNormalized')['SalaryNormalized'].median().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=salaries_by_location_median.values, y=salaries_by_location_median.index)
plt.title('Top 10 Locations by Median Salary')
plt.xlabel('Median Salary')
plt.ylabel('Location')
plt.show()


# In[56]:


# Highest paying companies
highest_paying_companies = df.groupby('Company')['SalaryNormalized'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=highest_paying_companies.values, y=highest_paying_companies.index)
plt.title('Top 10 Companies by Average Salary')
plt.xlabel('Average Salary')
plt.ylabel('Company')
plt.show()


# In[57]:


# Highest paying categories
salaries_by_category = df.groupby('Category')['SalaryNormalized'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=salaries_by_category.values, y=salaries_by_category.index)
plt.title('Top 10 Job Categories by Average Salary')    
plt.xlabel('Average Salary')
plt.ylabel('Job Category')
plt.show()


# In[58]:


# Boxplot of Salary by Category
plt.figure(figsize=(12,8))
sns.boxplot(x='Category', y='SalaryNormalized', data=df)
plt.title('Boxplot of Salary by Category')
plt.xlabel('Job Category')
plt.ylabel('SalaryNormalized')
plt.xticks(rotation=90)
plt.show()


# In[59]:


# Salary by SourceName
salaries_by_source = df.groupby('SourceName')['SalaryNormalized'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=salaries_by_source.values, y=salaries_by_source.index)
plt.title('Top 10 Sources by Average Salary')
plt.xlabel('Average Salary')
plt.ylabel('SourceName')    
plt.show()


# ## Phase 1: Baseline with categorical features

# In the first phase, we will create a baseline model using only categorical features. This will help us understand the performance of a simple model before incorporating text data. We will include columns: Category, Company, LocationNormalized, ContractType and ContractTime. As for now we will fill missing values with 'Unknown' value.
# 

# In[60]:


df = pd.read_csv(FILEPATH)
df.loc[df['Title'].isnull(), 'Title'] = 'Quality Improvement Manager'
cols_with_na = ["ContractType", "ContractTime", "Company", "SourceName"]
df[cols_with_na] = df[cols_with_na].fillna("Unknown")


# In[61]:


df_cat = df[['Category', 'Company', 'LocationNormalized', 'ContractType', 'ContractTime', 'SalaryNormalized']]
df_cat.head()


#  We need to convert categorical variables into numerical format.

# In[62]:


for col in df_cat.columns:
    print(f"{col}: {df_cat[col].nunique()} unique values")


# For columns Category, ContractType and ContractTime we will use One-Hot Encoding as they have a small number of unique values.
# For columns Company and LocationNormalized we will use Target Encoding as they have a large number of unique values.

# Before transforming our categorical variables, let's split the data into training, validation and test sets to avoid data leakage.

# ### Splitting the data into training, validation and test sets

# In[63]:


X = df_cat.drop(columns=['SalaryNormalized'])
y = df_cat['SalaryNormalized']

# Split the data into training (80%), validation (10%) and test (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)
print(f"Train set has {len(X_train)} entries")
print(f"Validation set has {len(X_valid)} entries")
print(f"Test set has {len(X_test)} entries")


# ### Creating custom Dataset class

# In[4]:


class SalaryDataset(Dataset):
    def __init__(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        """
        Custom PyTorch Dataset for salary prediction.

        This class handles the conversion of preprocessed features and scaled targets
        from numpy arrays or pandas data structures into PyTorch tensors.

        Args:
            X: Features for the dataset, can be a numpy array or a pandas DataFrame.
            y: Target values (salaries) for the dataset, can be a numpy array or a pandas Series.
        """
        self.X = torch.tensor(X.values, dtype=torch.float32) if isinstance(X, pd.DataFrame) else torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ### Preprocessing pipeline

# Before feeding the data into custom Dataset class, we will create a preprocessing pipeline that will handle the transformations of categorical variables and scaling of the target variable.

# In[65]:


one_hot_cols = ['Category', 'ContractType', 'ContractTime']
target_enc_cols = ['Company', 'LocationNormalized']
target_col = 'SalaryNormalized'

# One-hot encoding pipeline for columns with few unique values
one_hot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Target encoding pipeline for columns with many unique values, followed by scaling
target_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('target_enc', TargetEncoder()),
    ('scaler', StandardScaler())
])

# Combine both pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('onehot', one_hot_pipeline, one_hot_cols),
    ('target_scaled', target_pipeline, target_enc_cols)
])

# Fit and transform training data, transform validation and test data
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_valid_processed = preprocessor.transform(X_valid)
X_test_processed = preprocessor.transform(X_test)

# Scale target variable
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# Create PyTorch datasets
train_dataset = SalaryDataset(X_train_processed, y_train_scaled)
valid_dataset = SalaryDataset(X_valid_processed, y_valid_scaled)
test_dataset = SalaryDataset(X_test_processed, y_test_scaled)

# Save preprocessor and scaler for later use
joblib.dump(preprocessor, PREPROCESSORS_DIR + 'preprocessor.pkl')
joblib.dump(target_scaler, PREPROCESSORS_DIR + 'target_scaler.pkl')


# ### Creating data loaders

# In[66]:


batch_size = 32
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)


# In[67]:


# Check number of batches
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(valid_loader)}")
print(f"Number of test batches: {len(test_loader)}")


# In[68]:


for X_batch, y_batch in train_loader:
    print(f"Batch X shape: {X_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    print(X_batch)
    print(y_batch)
    break  # Just to check the first batch


# ### Building simple model

# In[69]:


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, dropout_prob: float = 0.2):
        """
        A simple feed-forward neural network for regression tasks.

        This model consists of two hidden layers with ReLU activation and dropout,
        followed by an output layer with a single neuron.

        Args:
            input_dim: The number of features in the input data.
            hidden_size: The number of neurons in the first hidden layer. The second
                         hidden layer will have half this number. Defaults to 64.
            dropout_prob: The dropout probability for the dropout layers. Defaults to 0.2.
        """
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ### Training the model

# #### Custom Early Stopping class

# In[5]:


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    This class monitors the validation loss and halts the training process if
    the loss does not decrease for a specified number of epochs. It can also
    restore the model's weights from the best-performing epoch.
    """
    def __init__(self, patience: int = 5, delta: float = 0, verbose: bool = False, restore_best_weights: bool = True):
        """
        Initializes the EarlyStopping instance.

        Args:
            patience: How many epochs to wait for a validation loss improvement before
                      stopping. Defaults to 5.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
                   Defaults to 0.
            verbose: If True, prints a message for each improvement and when early stopping
                     is triggered. Defaults to False.
            restore_best_weights: If True, the model's weights from the best-performing
                                  epoch are restored upon early stopping. Defaults to True.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.best_weights = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss: float, model: torch.nn.Module):
        """
        Checks the validation loss and updates the internal state.

        Args:
            val_loss: The current validation loss.
            model: The PyTorch model being trained.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f'Validation loss improved to {val_loss:.6f}.')
        else:
            self.no_improvement_count += 1
            if self.verbose:
                if self.no_improvement_count == 1:
                    print(f'No improvement in validation loss for {self.no_improvement_count} epoch.')
                else:
                    print(f'No improvement in validation loss for {self.no_improvement_count} epochs.')
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print('Restored best model weights.')
                if self.verbose:
                    print('Early stopping triggered.')


# #### Fuction to plot training and validation loss

# In[71]:


def plot_losses(
    train_losses: List[float], 
    valid_losses: List[float], 
    train_real_losses: Optional[List[float]] = None, 
    valid_real_losses: Optional[List[float]] = None
):
    """
    Plot training and validation losses.

    This function can plot either the scaled losses or both scaled and real-scale
    losses side-by-side, depending on whether the real-scale losses are provided.

    Args:
        train_losses: A list of scaled training loss values, one for each epoch.
        valid_losses: A list of scaled validation loss values, one for each epoch.
        train_real_losses: An optional list of real-scale training loss values.
                           Defaults to None.
        valid_real_losses: An optional list of real-scale validation loss values.
                           Defaults to None.
    """
    if train_real_losses is None or valid_real_losses is None:
        # Only plot scaled losses
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, color='blue', label='Train Loss', linewidth=2)
        ax.plot(epochs, valid_losses, color='orange', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (Scaled MSE)', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    else:
        # Plot both scaled and real losses
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        epochs = range(1, len(train_losses) + 1)

        # Plot 1: Scaled MSE
        axes[0].plot(epochs, train_losses, color='blue', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, valid_losses, color='orange', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss (Scaled MSE)', fontsize=11)
        axes[0].set_title('Scaled MSE Loss', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Real MSE
        axes[1].plot(epochs, train_real_losses, color='blue', label='Train Loss', linewidth=2)
        axes[1].plot(epochs, valid_real_losses, color='orange', label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss (Real MSE)', fontsize=11)
        axes[1].set_title('Real Scale MSE Loss', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# #### Training loop

# In[72]:


# Set random seed for reproducibility
set_seed()

# Define model hyperparameters
input_dim = X_train_processed.shape[1]
hidden_size = 128
dropout_prob = 0.2

# Initialize the model and move to device
model = SimpleRegressor(input_dim, hidden_size, dropout_prob).to(device)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load target scaler for inverse transformation
target_scaler = joblib.load(PREPROCESSORS_DIR + '/' + 'target_scaler.pkl')

# Training loop parameters
n_epochs = 30
patience = 4
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

# Training loop
for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss (scaled)
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate scaled loss
        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    # Average training losses
    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    # Validation loop
    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average validation losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# - `.reshape(-1, 1)` - StandardScaler expects 2D input, so reshape the 1D arrays
# - `.ravel()` - Flatten back to 1D after inverse transform for easier calculation
# - `.detach()` - Remove from computation graph (important in training loop)
# - `.cpu()` - Move to CPU for numpy operations

# #### Plotting training and validation loss

# In[73]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# #### Explore model parameters

# In[74]:


import torch.nn as nn

def print_model_parameters_summary(model: nn.Module):
    """
    Prints the parameter count for each named layer and the total sum.
    """
    total_params = 0

    # Print layer-wise details
    for name, param in model.named_parameters():
        num_params = param.numel()
        print(f"{name}: {num_params:,} parameters, trainable={param.requires_grad}")
        total_params += num_params

    # Print summed total
    print("-" * 40)
    print(f"Total Parameters: {total_params:,}")


# In[75]:


print_model_parameters_summary(model)


# #### Save the model

# In[76]:


model_name = 'cat_unk_bs32_adam_lrs_no_hid128_dr20'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# #### Evaluate on test set for later comparison

# In[77]:


baseline_models_test_scores = dict()


# In[78]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    baseline_models_test_scores[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# ### Hyperparameter tuning / other approaches

# #### Replacing missing values with most frequent values

# In[230]:


df = pd.read_csv(FILEPATH)  
df = df.drop(columns=['Id'])
df_cat = df[['Category', 'Company', 'LocationNormalized', 'ContractType', 'ContractTime', 'SalaryNormalized']]
X = df_cat.drop(columns=['SalaryNormalized'])
y = df_cat['SalaryNormalized']

# Split the data into training (80%), validation (10%) and test (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

one_hot_cols = ['Category', 'ContractType', 'ContractTime']
target_enc_cols = ['Company', 'LocationNormalized']
target_col = 'SalaryNormalized'

# One-hot encoding pipeline for columns with few unique values
one_hot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # fill missing values with most frequent
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Target encoding pipeline for columns with many unique values, followed by scaling
target_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # fill missing values with most frequent
    ('target_enc', TargetEncoder()),
    ('scaler', StandardScaler())
])

# Combine both pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('onehot', one_hot_pipeline, one_hot_cols),
    ('target_scaled', target_pipeline, target_enc_cols)
])

# Fit and transform training data, transform validation and test data
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_valid_processed = preprocessor.transform(X_valid)
X_test_processed = preprocessor.transform(X_test)

# Scale target variable
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# Create PyTorch datasets
train_dataset = SalaryDataset(X_train_processed, y_train_scaled)
valid_dataset = SalaryDataset(X_valid_processed, y_valid_scaled)
test_dataset = SalaryDataset(X_test_processed, y_test_scaled)

# Save preprocessor and scaler for later use
joblib.dump(preprocessor, PREPROCESSORS_DIR + 'preprocessor.pkl')
joblib.dump(target_scaler, PREPROCESSORS_DIR + 'target_scaler.pkl')

batch_size = 32
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)


# In[238]:


# Set random seed for reproducibility
set_seed()

# Define model architecture parameters
input_dim = X_train_processed.shape[1]
hidden_size = 128
dropout_prob = 0.2

# Initialize the regression model and move to device
model = SimpleRegressor(input_dim, hidden_size, dropout_prob).to(device)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load target scaler for inverse transformation
target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

# Training configuration
n_epochs = 30
patience = 4
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss for each epoch
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

# Training loop
for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss in scaled space
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate scaled loss
        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    # Calculate average training losses
    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    # Validation loop
    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Calculate average validation losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping condition
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[81]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[82]:


print_model_parameters_summary(model)


# In[83]:


model_name = 'cat_mf_bs32_adam_lrs_no_hid128_dr20'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[84]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    baseline_models_test_scores[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# For the next approach we will replace missing values in categorical columns with 'Unknown' value as it did give better results in the previous experiments.

# #### Trying batch size 64 

# In[85]:


one_hot_cols = ['Category', 'ContractType', 'ContractTime']
target_enc_cols = ['Company', 'LocationNormalized']
target_col = 'SalaryNormalized'

one_hot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

target_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('target_enc', TargetEncoder()),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('onehot', one_hot_pipeline, one_hot_cols),
    ('target_scaled', target_pipeline, target_enc_cols)
])

X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_valid_processed = preprocessor.transform(X_valid)
X_test_processed = preprocessor.transform(X_test)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

train_dataset = SalaryDataset(X_train_processed, y_train_scaled)
valid_dataset = SalaryDataset(X_valid_processed, y_valid_scaled)
test_dataset = SalaryDataset(X_test_processed, y_test_scaled)

joblib.dump(preprocessor, PREPROCESSORS_DIR + 'preprocessor.pkl')
joblib.dump(target_scaler, PREPROCESSORS_DIR + 'target_scaler.pkl')

batch_size = 64
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)


# In[86]:


# Set random seed for reproducibility
set_seed()

# Define model architecture parameters
input_dim = X_train_processed.shape[1]
hidden_size = 128
dropout_prob = 0.2

# Initialize the regression model and move to device
model = SimpleRegressor(input_dim, hidden_size, dropout_prob).to(device)

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load target scaler for inverse transformation
target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

# Training configuration
n_epochs = 30
patience = 4
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss for each epoch
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

# Training loop
for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss in scaled space
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate scaled loss
        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    # Calculate average training losses
    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    # Validation loop
    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Calculate average validation losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping condition
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[87]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[88]:


print_model_parameters_summary(model)


# In[89]:


model_name = 'cat_unk_bs64_adam_lrs_no_hid128_dr20'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[90]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    baseline_models_test_scores[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# #### Trying less neurons

# In[91]:


set_seed()

input_dim = X_train_processed.shape[1]
hidden_size = 64
dropout_prob = 0.2

model = SimpleRegressor(input_dim, hidden_size, dropout_prob).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 30
patience = 4
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[92]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[93]:


print_model_parameters_summary(model)


# In[94]:


model_name = 'cat_unk_bs32_adam_lrs_no_hid64_dr20'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[95]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    baseline_models_test_scores[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# #### Different optimizer

# In[96]:


# SGD optimizer with momentum
set_seed()

input_dim = X_train_processed.shape[1]
hidden_size = 128
dropout_prob = 0.2

model = SimpleRegressor(input_dim, hidden_size, dropout_prob).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 30
patience = 4
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[97]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[98]:


print_model_parameters_summary(model)


# In[99]:


model_name = 'cat_unk_bs64_sgd_lrs_no_hid128_dr20'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[100]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    baseline_models_test_scores[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# #### Setting learning rate scheduler

# In[101]:


set_seed()

input_dim = X_train_processed.shape[1]
hidden_size = 128
dropout_prob = 0.2

model = SimpleRegressor(input_dim, hidden_size, dropout_prob).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 30
patience = 4
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break

    # Step the learning rate scheduler
    lr_scheduler.step(valid_loss_scaled_avg)
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")


# In[102]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[103]:


print_model_parameters_summary(model)


# In[104]:


model_name = 'cat_unk_bs64_adam_lrs_hid128_dr20'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[105]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    baseline_models_test_scores[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# In[106]:


baseline_models_test_scores


# ### Evaluating models on the test set

# In[107]:


# Extract data
models = list(baseline_models_test_scores.keys())
mse_scaled = [baseline_models_test_scores[m]['Test MSE (scaled)'] for m in models]
mse_real = [baseline_models_test_scores[m]['Test MSE (real)'] for m in models]
rmse_real = [baseline_models_test_scores[m]['Test RMSE (real)'] for m in models]

# Sort by MSE real
sorted_indices = np.argsort(mse_real)
models_sorted = [models[i] for i in sorted_indices]
mse_scaled_sorted = [mse_scaled[i] for i in sorted_indices]
rmse_real_sorted = [rmse_real[i] for i in sorted_indices]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: MSE (scaled only)
ax1.bar(models_sorted, mse_scaled_sorted, alpha=0.8, color='blue')
ax1.set_xlabel('Model')
ax1.set_ylabel('MSE (scaled)')
ax1.set_xticklabels(models_sorted, rotation=45, ha='right')
ax1.set_title('Test MSE (Scaled)')
ax1.grid(alpha=0.3)

# Right: RMSE (real)
ax2.bar(models_sorted, rmse_real_sorted, alpha=0.8, color='green')
ax2.set_xlabel('Model')
ax2.set_ylabel('RMSE (real)')
ax2.set_xticklabels(models_sorted, rotation=45, ha='right')
ax2.set_title('Test RMSE (Real)')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# We can see that tuning hyperparameters for such a simple model with a few categorical features does not bring significant improvements.

# ## Phase 2: Incorporating text data

# Text data is present in the **`Title`** and **`FullDescription`** columns.  
# We need to preprocess these texts and convert them into a numerical format.
# 
# ### What we may try to extract from text data and what to do with it:
# - **Numbers** — e.g., “5 years experience”, “3+ years”, “2 years of experience”
# - **Programming languages** — e.g., “Python”, “Java”, “C++”, “JavaScript”
# - **Job seniority levels** — e.g., “Junior”, “Manager”, “Senior”, “Lead”
# - **Special characters** — e.g., “+” (C++), “#” (C#), “.” (.NET)
# - **Convert all text to lowercase**
# 
# ### What is not important and will be removed:
# - URLs  
# - Email addresses  
# - Asterisks (`***`) used for salary masking  
# - Excessive whitespaces  
# 
# ### To consider carefully:
# - **Stop words** — Usually removed, but in job descriptions they may be meaningful (e.g., “not”, “without”, “no experience required”).  
# - **Lemmatization / Stemming** — May alter words with different meanings (e.g., “developer” vs “development”), so should be applied with caution.
# 

# When we clean our text data we need to tokenize it.  The text needs to be broken down into individual words or sub-words, called tokens. This is a crucial step in NLP as it allows us to analyze the text at a granular level.

# Next step may be vectorization. This is the process of converting text data into numerical format that can be fed into machine learning models. There are several techniques for vectorization, including e.g.:
# 
# - TF-IDF (Term Frequency-Inverse Document Frequency)
# - Word Embeddings (e.g., Word2Vec, GloVe)
# 
# At start I will try TF-IDF as it is simple and effective for many NLP tasks.

# We may also try to include 'SourceName' column as it may contain some useful information about the job offer. It will be also target encoded as it has a large number of unique values.

# In[108]:


df.sample(5)


# In[109]:


# check wheter some descriptions have html tags
df['FullDescription'].str.contains('<.*?>').sum()
# no we don't have any html tags


# ### Custom Text Preprocessor and Tf-Idf Vectorizer

# #### Building Text Preprocessor

# In[6]:


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn compatible transformer for cleaning and preprocessing text data.

    This class handles various text cleaning steps such as converting to lowercase, 
    removing URLs, emails, HTML tags, and punctuation. It also includes optional steps 
    for removing numbers, stopwords, and lemmatization. It is designed to work with 
    pandas DataFrames containing 'Title' and 'FullDescription' columns.
    """
    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_stopwords: bool = False,
                 remove_numbers: bool = False,
                 lemmatize: bool = False):
        """
        Initializes the TextPreprocessor with various preprocessing options.

        Args:
            lowercase: If True, converts all text to lowercase. Defaults to True.
            remove_punctuation: If True, removes all punctuation from the text. 
                                Defaults to False.
            remove_stopwords: If True, removes common English stopwords. Defaults to False.
            remove_numbers: If True, removes all numbers from the text. Defaults to False.
            lemmatize: If True, reduces words to their base or root form. Defaults to False.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Cleans a single text string based on the initialized parameters.

        Args:
            text: The text string to be cleaned.

        Returns:
            The cleaned text string.
        """
        if pd.isnull(text) or text.strip() == '':
            return ''

        text = str(text)

        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags (just in case)
        text = re.sub(r'<.*?>', '', text)

        # Remove multiple asterisks (e.g., *****)
        text = re.sub(r'\*{2,}', ' ', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters
        text = re.sub(r'[^\s\w+#.+-]', '', text)

        # Remove numbers - optional
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation - optional
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenization
        tokens = text.split()

        # Remove stopwords - optional
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatization - optional
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back into a single string
        text = ' '.join(tokens)

        return text.strip()


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        if 'Title' in X_copy.columns:
            X_copy['Title'] = X_copy['Title'].apply(self.clean_text)

        if 'FullDescription' in X_copy.columns:
            X_copy['FullDescription'] = X_copy['FullDescription'].apply(self.clean_text)

        return X_copy

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.transform(X)


# In[111]:


tp = TextPreprocessor()
sample = df['FullDescription'].sample(1).values[0]
print("Original:")
print(sample)
print("\nCleaned:")
cleaned = tp.clean_text(sample)
print(cleaned)


# #### Building Tf-Idf Transformer

# In[7]:


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn compatible transformer for applying TF-IDF vectorization
    and optional Singular Value Decomposition (SVD) for dimensionality reduction.

    This transformer is designed to process a specific text column in a pandas DataFrame,
    convert it into a TF-IDF matrix, and can optionally reduce its dimensions using SVD.
    """
    def __init__(self,
                 text_column: str = 'Title',
                 max_features: int = 50,
                 use_svd: bool = False,
                 n_components: int = 10,
                 stop_words: Optional[List[str] | str] = None):
        """
        Initializes the TfidfTransformer.

        Args:
            text_column: The name of the DataFrame column containing the text data.
                         Defaults to 'Title'.
            max_features: The maximum number of features (tokens) to be considered by
                          the TfidfVectorizer. Defaults to 50.
            use_svd: If True, applies TruncatedSVD for dimensionality reduction.
                     Defaults to False.
            n_components: The number of components to keep after SVD. This is only
                          used if `use_svd` is True. Defaults to 10.
            stop_words: A list of stop words or a string indicating a language
                        (e.g., 'english'). Passed directly to TfidfVectorizer.
                        Defaults to None.
        """
        self.text_column = text_column
        self.max_features = max_features
        self.use_svd = use_svd
        self.n_components = n_components
        self.stop_words = stop_words

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # Underscore suffix for fitted attributes (sklearn convention)
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=(1, 2),
            min_df=5
        )

        text_data = X[self.text_column].fillna('')
        tfidf_matrix = self.vectorizer_.fit_transform(text_data)

        if self.use_svd:
            self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=RANDOM_SEED)
            self.svd_.fit(tfidf_matrix)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        # Check for fitted attribute
        if not hasattr(self, 'vectorizer_'):
            raise RuntimeError("This TfidfTransformer instance is not fitted yet. Call 'fit' first.")

        text_data = X[self.text_column].fillna('')
        tfidf_matrix = self.vectorizer_.transform(text_data)

        if self.use_svd:
            tfidf_matrix = self.svd_.transform(tfidf_matrix)
            return tfidf_matrix  # Already a numpy array
        else:
            return tfidf_matrix.toarray()


# #### Complete preprocessing pipeline

# In[113]:


df = pd.read_csv(FILEPATH)
X = df[['Title', 'FullDescription', 'Category', 'Company', 'LocationNormalized', 'ContractType', 'ContractTime', 'SourceName']]
X['Title'] = X['Title'].fillna('Quality Improvement Manager') # based on EDA
y = df['SalaryNormalized']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)
print(f"Train set has {len(X_train)} entries")
print(f"Validation set has {len(X_valid)} entries")
print(f"Test set has {len(X_test)} entries")


# In[114]:


text_columns = ['Title', 'FullDescription']
categorical_columns = ['Category', 'ContractType', 'ContractTime']
high_cardinality_columns = ['Company', 'LocationNormalized', 'SourceName']

one_hot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

target_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), 
    ('target_enc', TargetEncoder()),
    ('scaler', StandardScaler())
])

title_pipeline = make_pipeline(
    TextPreprocessor(lowercase=True, remove_punctuation=False, 
                    remove_stopwords=False, remove_numbers=False, lemmatize=False),
    TfidfTransformer(text_column='Title', max_features=50, use_svd=False, stop_words='english')
)

desc_pipeline = make_pipeline(
    TextPreprocessor(lowercase=True, remove_punctuation=False, 
                    remove_stopwords=False, remove_numbers=False, lemmatize=False),
    TfidfTransformer(text_column='FullDescription', max_features=800, use_svd=False, stop_words='english')
)

preprocessor_with_text = ColumnTransformer([
    ('title_tfidf', title_pipeline, ['Title']),
    ('desc_tfidf', desc_pipeline, ['FullDescription']),
    ('onehot', one_hot_pipeline, categorical_columns),
    ('target_scaled', target_pipeline, high_cardinality_columns)
])


target_scaler = StandardScaler() 
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()  
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()      
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()      

X_train_processed = preprocessor_with_text.fit_transform(X_train, y_train)  
X_valid_processed = preprocessor_with_text.transform(X_valid)               
X_test_processed = preprocessor_with_text.transform(X_test)                 

train_dataset = SalaryDataset(X_train_processed, y_train_scaled)
valid_dataset = SalaryDataset(X_valid_processed, y_valid_scaled)
test_dataset = SalaryDataset(X_test_processed, y_test_scaled)

# Save the preprocessor and target scaler for future use
joblib.dump(preprocessor_with_text, PREPROCESSORS_DIR + 'preprocessor_with_text.pkl')
joblib.dump(target_scaler, PREPROCESSORS_DIR + 'target_scaler.pkl')


# #### Create dataloaders

# In[115]:


batch_size = 64
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)


# In[116]:


for X_batch, y_batch in train_loader:
    print(f"Batch X shape: {X_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    print(X_batch)
    print(y_batch)
    break  # Just to check the first batch


# #### Training the model with text data

# In[117]:


set_seed()

input_dim = X_train_processed.shape[1]
hidden_size = 128
dropout_prob = 0.3 #higher to prevent overfitting

model = SimpleRegressor(input_dim, hidden_size, dropout_prob).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 25
patience = 3
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[118]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[119]:


print_model_parameters_summary(model)


# #### Save the model

# In[120]:


model_name = 'tfidf_800_50_sr_unk_bs64_adam_lrs_no_hid128_dr30'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# #### Evaluate on the test set for later comparison

# In[121]:


fulldata_best_models = dict()  # To store test scores of the best models


# In[122]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    fulldata_best_models[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# #### Removing redundancy

# **TextPreprocessor does:**
# - Lowercase 
# - Remove URLs, emails 
# - Remove special chars
# - Remove stopwords (optional)
# 
# **TfidfVectorizer ALSO does**
# - Lowercase (by default)
# - Remove stopwords (optional)
# - Tokenization
# - Filters by min_df

# So if we know that despite TextPreprocessor we will also use TfidfVectorizer, we can simplify TextPreprocessor to only remove URLs and emails.

# In[10]:


class MinimalTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn compatible transformer for minimal text cleaning.

    This class performs a streamlined set of text cleaning operations that are not
    typically handled by TF-IDF vectorizers, such as removing URLs, email addresses,
    HTML tags, and multiple asterisks. It is designed to work with pandas DataFrames
    containing 'Title' and 'FullDescription' columns.
    """

    def clean_text(self, text: str) -> str:
        """
        Cleans a single text string by removing specific patterns.

        Args:
            text: The text string to be cleaned.

        Returns:
            The cleaned text string.
        """
        if pd.isnull(text):
            return ''
        text = str(text).strip()
        if text == '':
            return ''

        # Remove URLs, emails, HTML, asterisks
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\*{2,}', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if 'Title' in X_copy.columns:
            X_copy['Title'] = X_copy['Title'].apply(self.clean_text)
        if 'FullDescription' in X_copy.columns:
            X_copy['FullDescription'] = X_copy['FullDescription'].apply(self.clean_text)
        return X_copy


# In[124]:


text_columns = ['Title', 'FullDescription']
categorical_columns = ['Category', 'ContractType', 'ContractTime']
high_cardinality_columns = ['Company', 'LocationNormalized', 'SourceName']

# Shared text preprocessor 
shared_text_prep = MinimalTextPreprocessor()

# Title pipeline
title_pipeline = make_pipeline(
    TfidfTransformer(text_column='Title', max_features=50)
)

# Description pipeline
desc_pipeline = make_pipeline(
    TfidfTransformer(text_column='FullDescription', max_features=800)
)

# Categorical pipeline
one_hot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# High cardinality pipeline
target_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), 
    ('target_enc', TargetEncoder()),
    ('scaler', StandardScaler())
])

# Clean text first (once for all text columns)
X_train_clean = shared_text_prep.fit_transform(X_train)
X_valid_clean = shared_text_prep.transform(X_valid)
X_test_clean = shared_text_prep.transform(X_test)

# Feature extraction
preprocessor = ColumnTransformer([
    ('title_tfidf', title_pipeline, ['Title']),
    ('desc_tfidf', desc_pipeline, ['FullDescription']),
    ('onehot', one_hot_pipeline, categorical_columns),
    ('target_scaled', target_pipeline, high_cardinality_columns)
])

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train_clean, y_train)
X_valid_processed = preprocessor.transform(X_valid_clean)
X_test_processed = preprocessor.transform(X_test_clean)

# Scale target
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# Create datasets
train_dataset = SalaryDataset(X_train_processed, y_train_scaled)
valid_dataset = SalaryDataset(X_valid_processed, y_valid_scaled)
test_dataset = SalaryDataset(X_test_processed, y_test_scaled)

# Save for later
joblib.dump(shared_text_prep, PREPROCESSORS_DIR + 'text_preprocessor.pkl')
joblib.dump(preprocessor, PREPROCESSORS_DIR + 'preprocessor.pkl')
joblib.dump(target_scaler, PREPROCESSORS_DIR + 'target_scaler.pkl')

print(f"Final feature shape: {X_train_processed.shape}")


# - Now we can just repeat the training loop as before
# - We may try to experiment with:
#     - Different optimizers
#     - Different learning rates (for example using learning rate schedulers)
#     - Different batch sizes
#     - Using stop words in TfidfVectorizer
#     - Using SVD to reduce dimensionality of text features
#     - Number of features in TfidfVectorizer

# #### Building more complex model with text data

# Now I will add batch normalization and **try** define learning rate scheduler.

# In[11]:


import torch
import torch.nn as nn

class SimpleRegressorWithNormalization(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, dropout_prob: float = 0.3):
        """
        A simple feed-forward neural network for regression with Batch Normalization.

        This model includes Batch Normalization layers after each linear layer and before the
        activation function, which helps to stabilize and accelerate training.

        Args:
            input_dim: The number of features in the input data.
            hidden_size: The number of neurons in the first hidden layer. The second
                         hidden layer will have half this number. Defaults to 64.
            dropout_prob: The dropout probability for the dropout layers. Defaults to 0.3.
        """
        super(SimpleRegressorWithNormalization, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# In[126]:


# Number of parameters
input_dim = X_train_processed.shape[1]
print(f"Input dimension: {input_dim}")
model = SimpleRegressorWithNormalization(input_dim, hidden_size=128, dropout_prob=0.3)
for name, param in model.named_parameters():
    print(f"{name}: {param.numel()} parameters, trainable={param.requires_grad}")

print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")


# In[127]:


batch_size = 64
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)


# In[128]:


set_seed()

input_dim = X_train_processed.shape[1]
hidden_size = 128
dropout_prob = 0.3
patience_scheduler = 2
lr = 0.001

model = SimpleRegressorWithNormalization(input_dim, hidden_size, dropout_prob).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
### scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 25
patience = 3
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Step the scheduler
    ### scheduler.step(valid_loss_scaled_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[129]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[130]:


print_model_parameters_summary(model)


# In[131]:


model_name = 'tfidf_800_50_srbn_unk_bs64_adam_lrs_no_hid128_dr30'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[132]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    fulldata_best_models[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# ### Using Pretrained Embeddings

# Firstly I will try with sentence-transformers library.

# How to use it? - Code from the documentation:

# In[133]:


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])


# ### Pipeline for using pretrained embeddings

# In[12]:


class TextEmbedder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to generate text embeddings using a pre-trained model.

    This class leverages the `sentence-transformers` library to convert text data into
    numerical vector representations (embeddings) suitable for machine learning models.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L12-v2'):
        """
        Initializes the TextEmbedder.

        Args:
            model_name: The name of the pre-trained SentenceTransformer model to use.
                        Defaults to 'all-MiniLM-L12-v2'.
        """
        self.model_name = model_name
        self.model = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # The model is pre-trained, so we just load it here.
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Convert input to a list of strings and handle potential NaNs.
        X = [str(x) if x is not None else '' for x in X.iloc[:, 0].tolist()]

        # Generate the embeddings.
        embeddings = self.model.encode(X, show_progress_bar=False)
        return embeddings

    def _check_n_features(self, X, reset):
        """Ensure compatibility with sklearn's feature validation"""
        pass


# In[135]:


categorical_columns = ['Category', 'ContractType', 'ContractTime']
high_cardinality_columns = ['Company', 'LocationNormalized', 'SourceName']

one_hot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

target_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), 
    ('target_enc', TargetEncoder()), 
    ('scaler', StandardScaler())
])

# New embedding pipelines
title_embedding_pipeline = Pipeline([
    ('embedder', TextEmbedder())
])

desc_embedding_pipeline = Pipeline([
    ('embedder', TextEmbedder())
])

preprocessor_with_embeddings = ColumnTransformer([
    ('title_embeddings', title_embedding_pipeline, ['Title']),
    ('desc_embeddings', desc_embedding_pipeline, ['FullDescription']),
    ('onehot', one_hot_pipeline, categorical_columns),
    ('target_scaled', target_pipeline, high_cardinality_columns)
])

# Fit and transform
X_train_processed = preprocessor_with_embeddings.fit_transform(X_train, y_train)
X_valid_processed = preprocessor_with_embeddings.transform(X_valid)
X_test_processed = preprocessor_with_embeddings.transform(X_test)

# Scale target
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# Create datasets
train_dataset = SalaryDataset(X_train_processed, y_train_scaled)
valid_dataset = SalaryDataset(X_valid_processed, y_valid_scaled)
test_dataset = SalaryDataset(X_test_processed, y_test_scaled)

# Save for later
joblib.dump(preprocessor_with_embeddings, PREPROCESSORS_DIR + 'preprocessor_with_embeddings.pkl')
joblib.dump(target_scaler, PREPROCESSORS_DIR + 'target_scaler.pkl')

print(f"Final feature shape: {X_train_processed.shape}")


# ### Training simple model with embeddings

# In[136]:


batch_size = 64
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)


# In[137]:


set_seed()

input_dim = X_train_processed.shape[1]
hidden_size = 128
dropout_prob = 0.3
patience_scheduler = 2
lr = 0.001

model = SimpleRegressorWithNormalization(input_dim, hidden_size, dropout_prob).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
### scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 25
patience = 4
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)
print_mae = True  # MAE was a main score in the competition

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Step the scheduler
    ### scheduler.step(valid_loss_scaled_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[138]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[139]:


print_model_parameters_summary(model)


# In[140]:


model_name = 'emb_srbn_unk_bs64_adam_lrs_no_hid128_dr30'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[141]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    fulldata_best_models[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# ### Building deeper model for embeddings and tabular data

# I will add one more layer, more neurons and try with batch normalization before and after activation functions.

# In[13]:


class IntegratedNN(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256, dropout_prob: float = 0.3, batch_norm_before_activation: bool = True):
        """
        An integrated feed-forward neural network with configurable batch normalization placement.

        This model is a multi-layered perceptron designed for regression, featuring
        batch normalization and dropout for improved training stability and generalization.
        The order of batch normalization and activation can be specified.

        Args:
            input_dim: The number of input features.
            hidden_size: The number of neurons in the first hidden layer. Subsequent
                         layers will have a decreasing number of neurons. Defaults to 256.
            dropout_prob: The dropout probability applied after each activation. Defaults to 0.3.
            batch_norm_before_activation: If True, batch normalization is applied before
                                          the ReLU activation. If False, it is applied after.
                                          Defaults to True.
        """
        super(IntegratedNN, self).__init__()
        self.batch_norm_before_activation = batch_norm_before_activation
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2 // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2 // 2)
        self.final_dropout = nn.Dropout(p=min(0.2, dropout_prob))
        self.fc4 = nn.Linear(hidden_size // 2 // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_norm_before_activation:
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.final_dropout(x)
            x = self.fc4(x)
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.bn3(x)
            x = self.final_dropout(x)
            x = self.fc4(x)
        return x


# In[143]:


# Number of parameters
input_dim = X_train_processed.shape[1]
print(f"Input dimension: {input_dim}")
model = IntegratedNN(input_dim, hidden_size=256, dropout_prob=0.3)
for name, param in model.named_parameters():
    print(f"{name}: {param.numel()} parameters, trainable={param.requires_grad}")

print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")


# In[144]:


input_dim = X_train_processed.shape[1]
hidden_size = 256
dropout_prob = 0.3
patience_scheduler = 2
lr = 0.001
batch_norm_before_activation = True

model = IntegratedNN(input_dim, hidden_size, dropout_prob, batch_norm_before_activation).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
### scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 25
patience = 5
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Step the scheduler
    ### scheduler.step(valid_loss_scaled_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[145]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[146]:


model_name = 'emb_int_unk_bs64_adam_lrs_no_hid256_dr30'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[147]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    fulldata_best_models[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# Now change order of batch normalization and activation functions.

# In[148]:


set_seed()

input_dim = X_train_processed.shape[1]
hidden_size = 256
dropout_prob = 0.3
patience_scheduler = 2
lr = 0.001
batch_norm_before_activation = False

model = IntegratedNN(input_dim, hidden_size, dropout_prob, batch_norm_before_activation).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
### scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 25
patience = 5
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * X_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(train_loader.dataset)
    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()
            loss = loss_fn(predictions, y_batch)
            valid_loss_scaled += loss.item() * X_batch.size(0)

            # Compute MSE in real scale
            predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
            mse_real = np.mean((predictions_real - y_batch_real) ** 2)
            valid_loss_real += mse_real * X_batch.size(0)

    # Average losses
    valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)
    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Step the scheduler
    ### scheduler.step(valid_loss_scaled_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# MSE for validation is high however MAE is relatively low.

# In[149]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[150]:


print_model_parameters_summary(model)


# In[151]:


model_name = 'emb_int_unk_bs64_adam_lrs_no_hid256_dr30_batchnorm_after'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[152]:


model.eval()

with torch.no_grad():
    test_loss_scaled = 0.0
    test_loss_real = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * X_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        test_loss_real += mse_real * X_batch.size(0)

    # Average losses
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)
    test_loss_real_avg = test_loss_real / len(test_loader.dataset)

    fulldata_best_models[model_name] = {
        'Test MSE (scaled)': test_loss_scaled_avg,
        'Test MSE (real)': test_loss_real_avg,
        'Test RMSE (real)': np.sqrt(test_loss_real_avg)
    }


# Poor performance on validation.

# ### Building Multi-Input Model with Embeddings and Categorical Features

# At first we need to define which columns are representation of embeddings and which are categorical features.

# In[153]:


# Get the fitted transformers to find their dimensions
title_embedder = preprocessor_with_embeddings.named_transformers_['title_embeddings']['embedder']
desc_embedder = preprocessor_with_embeddings.named_transformers_['desc_embeddings']['embedder']
onehot_encoder = preprocessor_with_embeddings.named_transformers_['onehot']['onehot']

# Calculate the dimensions of each feature group
embedding_dim = title_embedder.model.get_sentence_embedding_dimension()  # Should be 384
print(f"Embedding dimension: {embedding_dim}") # two times 384 because we have title and description
title_dim = embedding_dim
desc_dim = embedding_dim
onehot_dim = onehot_encoder.get_feature_names_out().shape[0]  # Total one-hot features
print(f"One-hot dimension: {onehot_dim}")
target_dim = len(high_cardinality_columns) # 3
print(f"Target encoded dimension: {target_dim}")

# Define the start and end indices for each slice
# Note: The order is crucial and must match the ColumnTransformer.
title_end_idx = title_dim
desc_end_idx = title_dim + desc_dim
onehot_end_idx = desc_end_idx + onehot_dim
target_end_idx = onehot_end_idx + target_dim

# Now you can slice the processed data
embeddings_features = X_train_processed[:, :desc_end_idx]
tabular_features = X_train_processed[:, desc_end_idx:]

print(f"Total embeddings shape: {embeddings_features.shape}")
print(f"Total tabular shape: {tabular_features.shape}")


# In[154]:


# Now we can set tabular_start_index
tabular_start_index = desc_end_idx


# Now we need to define new custom Dataset class that will handle both types of inputs.

# In[18]:


class MultiInputDataset(Dataset):
    def __init__(self, X_processed: Union[np.ndarray, pd.DataFrame], y_scaled: np.ndarray, tabular_start_index: int):
        """
        Custom PyTorch Dataset for handling multiple input types (e.g., embeddings and tabular data).

        This class splits the preprocessed feature matrix into its constituent parts
        (e.g., text embeddings and tabular features) and prepares them as PyTorch tensors.

        Args:
            X_processed: The combined feature matrix from a `ColumnTransformer`, expected
                         to be a numpy array or pandas DataFrame.
            y_scaled: The scaled target values, as a numpy array.
            tabular_start_index: The column index where the tabular features begin.
                                 Features before this index are considered embeddings.
        """
        # Ensure X_processed is a numpy array for consistent tensor conversion
        if isinstance(X_processed, pd.DataFrame):
            X_processed = X_processed.values

        self.X_processed = torch.from_numpy(X_processed).float()
        self.y_scaled = torch.from_numpy(y_scaled).float()
        self.tabular_start_index = tabular_start_index

    def __len__(self) -> int:
        return len(self.X_processed)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample by its index, splitting features into embeddings and tabular data.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing three tensors: the embeddings, the tabular features,
            and the corresponding target value for the given index.
        """
        embeddings = self.X_processed[idx, :self.tabular_start_index]
        tabular_features = self.X_processed[idx, self.tabular_start_index:]

        target = self.y_scaled[idx]

        return embeddings, tabular_features, target


# In[156]:


multi_input_train_dataset = MultiInputDataset(X_train_processed, y_train_scaled, tabular_start_index)
multi_input_valid_dataset = MultiInputDataset(X_valid_processed, y_valid_scaled, tabular_start_index)
multi_input_test_dataset = MultiInputDataset(X_test_processed, y_test_scaled, tabular_start_index)


# In[157]:


# Try get a batch
for embeddings, tabular, target in DataLoader(multi_input_train_dataset, batch_size=4, shuffle=True):
    print(f"Embeddings shape: {embeddings.shape}")  # Should be (batch_size, embedding_dim*2)
    print(f"Example embeddings: {embeddings}")
    print(f"Tabular shape: {tabular.shape}")        # Should be (batch_size, tabular_dim)
    print(f"Example tabular: {tabular}")
    print(f"Target shape: {target.shape}")          # Should be (batch_size, 1)
    print(f"Example target: {target}")
    break


# In[158]:


batch_size = 64 
num_workers = 0

multi_input_train_loader = DataLoader(multi_input_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
multi_input_valid_loader = DataLoader(multi_input_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
multi_input_test_loader = DataLoader(multi_input_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)


# ### Building multi-input model

# In[19]:


class MultiInputNN(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 tabular_dim: int, 
                 embedding_hidden: List[int] = [256, 128], 
                 tabular_hidden: List[int] = [64, 32], 
                 combined_hidden: List[int] = [128, 64], 
                 dropout_prob: float = 0.3):
        """
        A neural network designed to handle multiple input types, such as text embeddings and tabular data.

        The network consists of three main components: a sub-network for processing embeddings,
        a sub-network for processing tabular features, and a combined network that
        concatenates their outputs to produce a final prediction. Each sub-network
        uses a sequence of linear layers, batch normalization, ReLU activation, and dropout.

        Args:
            embedding_dim: The dimension of the input text embeddings.
            tabular_dim: The number of features in the tabular data.
            embedding_hidden: A list of integers representing the number of neurons in
                              the hidden layers of the embedding sub-network. Defaults to [256, 128].
            tabular_hidden: A list of integers representing the number of neurons in
                            the hidden layers of the tabular sub-network. Defaults to [64, 32].
            combined_hidden: A list of integers representing the number of neurons in
                             the hidden layers of the combined network. Defaults to [128, 64].
            dropout_prob: The dropout probability applied in all dropout layers. Defaults to 0.3.
        """
        super(MultiInputNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.tabular_dim = tabular_dim
        self.embedding_hidden = embedding_hidden
        self.tabular_hidden = tabular_hidden
        self.dropout_prob = dropout_prob

        # Embedding layers
        embedding_layers = []
        input_size = embedding_dim
        for hidden_size in embedding_hidden:
            embedding_layers.append(nn.Linear(input_size, hidden_size))
            embedding_layers.append(nn.BatchNorm1d(hidden_size))
            embedding_layers.append(nn.ReLU())
            embedding_layers.append(nn.Dropout(p=dropout_prob))
            input_size = hidden_size

        self.embedding_net = nn.Sequential(*embedding_layers)

        # Tabular layers
        tabular_layers = []
        input_size = tabular_dim
        for hidden_size in tabular_hidden:
            tabular_layers.append(nn.Linear(input_size, hidden_size))
            tabular_layers.append(nn.BatchNorm1d(hidden_size))
            tabular_layers.append(nn.ReLU())
            tabular_layers.append(nn.Dropout(p=dropout_prob))
            input_size = hidden_size

        self.tabular_net = nn.Sequential(*tabular_layers)

        # Combined layers
        combined_input_size = embedding_hidden[-1] + tabular_hidden[-1]
        combined_layers = []
        for hidden_size in combined_hidden:
            combined_layers.append(nn.Linear(combined_input_size, hidden_size))
            combined_layers.append(nn.BatchNorm1d(hidden_size))
            combined_layers.append(nn.ReLU())
            combined_layers.append(nn.Dropout(p=dropout_prob))
            combined_input_size = hidden_size

        combined_layers.append(nn.Linear(combined_input_size, 1))  # Final output layer
        self.combined_net = nn.Sequential(*combined_layers)

    def forward(self, embeddings: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_net(embeddings)
        tabular = self.tabular_net(tabular)
        combined = torch.cat((embeddings, tabular), dim=1)
        output = self.combined_net(combined)
        return output


# #### Training multi-input model

# In[160]:


set_seed()

input_dim = X_train_processed.shape[1]
embedding_dim = 2 * 384 # 2 * 384 because we have title and description = 768
tabular_dim = input_dim - embedding_dim

print(f"Embedding dim: {embedding_dim}, Tabular dim: {tabular_dim}")

dropout_prob = 0.3 
lr = 0.001

model = MultiInputNN(embedding_dim=embedding_dim, tabular_dim=tabular_dim, dropout_prob=dropout_prob).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

target_scaler = joblib.load(PREPROCESSORS_DIR + 'target_scaler.pkl')

n_epochs = 25
patience = 3
delta = 0.001
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True, restore_best_weights=True)

# Lists to store training and validation loss
train_losses_scaled = []
valid_losses_scaled = []
train_losses_real = []
valid_losses_real = []

for epoch in range(n_epochs):
    model.train()
    train_loss_scaled = 0.0
    train_loss_real = 0.0
    for embeddings_batch, tabular_batch, y_batch in multi_input_train_loader:
        embeddings_batch, tabular_batch, y_batch = embeddings_batch.to(device), tabular_batch.to(device), y_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(embeddings_batch, tabular_batch).squeeze()

        # Compute loss
        loss = loss_fn(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss_scaled += loss.item() * embeddings_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        train_loss_real += mse_real * embeddings_batch.size(0)

    train_loss_scaled_avg = train_loss_scaled / len(multi_input_train_loader.dataset)
    train_loss_real_avg = train_loss_real / len(multi_input_train_loader.dataset)

    train_losses_scaled.append(train_loss_scaled_avg)
    train_losses_real.append(train_loss_real_avg)

    model.eval()
    valid_loss_scaled = 0.0
    valid_loss_real = 0.0
    for embeddings_batch, tabular_batch, y_batch in multi_input_valid_loader:
        embeddings_batch, tabular_batch, y_batch = embeddings_batch.to(device), tabular_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(embeddings_batch, tabular_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        valid_loss_scaled += loss.item() * embeddings_batch.size(0)

        # Compute MSE in real scale
        predictions_real = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).ravel()
        y_batch_real = target_scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1)).ravel()
        mse_real = np.mean((predictions_real - y_batch_real) ** 2)
        valid_loss_real += mse_real * embeddings_batch.size(0)

    valid_loss_scaled_avg = valid_loss_scaled / len(multi_input_valid_loader.dataset)
    valid_loss_real_avg = valid_loss_real / len(multi_input_valid_loader.dataset)

    valid_losses_scaled.append(valid_loss_scaled_avg)
    valid_losses_real.append(valid_loss_real_avg)

    # Print epoch results
    print(f'Epoch {epoch+1}/{n_epochs}:')
    print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {np.sqrt(train_loss_real_avg):.2f}')
    print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {np.sqrt(valid_loss_real_avg):.2f}')

    # Check early stopping
    early_stopping.check_early_stop(valid_loss_scaled_avg,  model)
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch+1}") 
        break


# In[161]:


plot_losses(train_losses_scaled, valid_losses_scaled, train_losses_real, valid_losses_real)


# In[162]:


print_model_parameters_summary(model)


# In[163]:


model_name = 'emb_multi_unk_bs64_adam_lrs_no_hid256_dr30'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[164]:


model.eval()

test_loss_scaled = 0.0
all_predictions_real = []
all_targets_real = []

with torch.no_grad():
    for embeddings_batch, tabular_batch, y_batch in multi_input_test_loader:
        embeddings_batch = embeddings_batch.to(device)
        tabular_batch = tabular_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass (multi-input)
        predictions = model(embeddings_batch, tabular_batch).squeeze()

        # Accumulate loss in scaled domain
        loss = loss_fn(predictions, y_batch)
        test_loss_scaled += loss.item() * embeddings_batch.size(0)

        # Store predictions and targets for real-scale metrics
        all_predictions_real.extend(target_scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).ravel())
        all_targets_real.extend(target_scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1)).ravel())

# Average loss over the entire dataset
test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)

# Calculate real-scale metrics over the entire dataset
all_predictions_real = np.array(all_predictions_real)
all_targets_real = np.array(all_targets_real)

mse_real = np.mean((all_predictions_real - all_targets_real) ** 2)
rmse_real = np.sqrt(mse_real)

fulldata_best_models[model_name] = {
    'Test MSE (scaled)': test_loss_scaled_avg,
    'Test MSE (real)': mse_real,
    'Test RMSE (real)': rmse_real
}


# In[165]:


def evaluate_multi_input_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    target_scaler,
    device: torch.device,
    loss_fn_str: str = 'mse',
    model_name: str = None,
    results_dict: Dict = None
) -> Dict:
    """
    Evaluates a multi-input PyTorch regression model and stores results in a dictionary.

    Args:
        model: The trained PyTorch model (expects multi-input).
        test_loader: DataLoader for the test data (MultiInputDataset).
        target_scaler: The fitted scaler for inverse transforming predictions.
        device: torch device (cuda or cpu).
        loss_fn_str: The metric to calculate ('mse' or 'mae').
        model_name: Name/identifier for the model (optional).
        results_dict: Dictionary to store results (optional, will update in-place).

    Returns:
        A dictionary containing the calculated metrics.
    """
    model.eval()

    # Select the appropriate loss function
    if loss_fn_str == 'mse':
        loss_function = nn.MSELoss()
    elif loss_fn_str == 'mae':
        loss_function = nn.L1Loss()
    else:
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Metrics for evaluation
    test_loss_scaled = 0.0
    all_predictions_real = []
    all_targets_real = []

    with torch.no_grad():
        for embeddings_batch, tabular_batch, y_batch in test_loader:
            embeddings_batch = embeddings_batch.to(device)
            tabular_batch = tabular_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass (multi-input)
            predictions = model(embeddings_batch, tabular_batch).squeeze()

            # Accumulate loss in scaled domain
            loss = loss_function(predictions, y_batch)
            test_loss_scaled += loss.item() * embeddings_batch.size(0)

            # Store predictions and targets for real-scale metrics
            all_predictions_real.extend(target_scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).ravel())
            all_targets_real.extend(target_scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1)).ravel())

    # Average loss over the entire dataset
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)

    # Calculate real-scale metrics over the entire dataset
    all_predictions_real = np.array(all_predictions_real)
    all_targets_real = np.array(all_targets_real)

    metrics = {}
    if loss_fn_str == 'mse':
        mse_real = np.mean((all_predictions_real - all_targets_real) ** 2)
        metrics['Test MSE (scaled)'] = test_loss_scaled_avg
        metrics['Test MSE (real)'] = mse_real
        metrics['Test RMSE (real)'] = np.sqrt(mse_real)
    elif loss_fn_str == 'mae':
        mae_real = np.mean(np.abs(all_predictions_real - all_targets_real))
        metrics['Test MAE (scaled)'] = test_loss_scaled_avg
        metrics['Test MAE (real)'] = mae_real

    # Store in results_dict if provided
    if results_dict is not None and model_name is not None:
        results_dict[model_name] = metrics

    return metrics


# ## Combine everything together

# **List of models I have created:**
# 
# - `SimpleRegressor`
# - `SimpleRegressorWithNormalization`
# - `IntegratedNN`
# - `MultiInputNN`
# 
# 

# ### Function to load data, select features and split into train, val, test sets

# In[20]:


def load_and_split_data(
    file_path: str = 'Train_rev1.csv',
    test_size: float = 0.2,
    valid_size: float = 0.5,
    random_state: int = RANDOM_SEED,
    log: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Loads data from a CSV, handles missing value in 'Title', and splits it into
    training, validation, and test sets.

    Args:
        file_path: Path to the raw CSV data file.
        test_size: Proportion of the dataset to include in the test split.
        valid_size: Proportion of the temporary split to include in the
                    validation split.
        random_state: Seed for reproducible train/test splits.
        log: Whether to apply log transformation to the target variable.
    Returns:
        A tuple containing:
        - X_train (pd.DataFrame): Training features.
        - X_valid (pd.DataFrame): Validation features.
        - X_test (pd.DataFrame): Test features.
        - y_train (pd.Series): Training target.
        - y_valid (pd.Series): Validation target.
        - y_test (pd.Series): Test target.
    """
    # Load data from the specified path
    df = pd.read_csv(file_path)

    # Separate features (X) and target (y)
    X = df[['Title', 'FullDescription', 'Category', 'Company', 
            'LocationNormalized', 'ContractType', 'ContractTime', 'SourceName']]
    if log:
        y = np.log1p(df['SalaryNormalized'])  # log(1 + x) to handle zero salaries if any
    else:
        y = df['SalaryNormalized']

    # Impute missing value in 'Title' based on EDA findings.
    X.loc[X['Title'].isnull(), 'Title'] = 'Quality Improvement Manager' 

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=valid_size, random_state=random_state
    )

    print(f"Train set has {len(X_train)} entries")
    print(f"Validation set has {len(X_valid)} entries")
    print(f"Test set has {len(X_test)} entries")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


# ### Function to preprocess data, create datasets and dataloaders for pretrained embeddings

# In[21]:


def preprocess_salary_data_with_embeddings(
    X_train: pd.DataFrame, 
    X_valid: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series, 
    y_valid: pd.Series, 
    y_test: pd.Series,
    categorical_columns: Optional[List[str]] = ['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns: Optional[List[str]] = ['Company', 'LocationNormalized', 'SourceName'],
    title_column: str = 'Title',
    desc_column: str = 'FullDescription',
    embedding_model_name: str = 'all-MiniLM-L12-v2',
    batch_size: int = 64,
    num_workers: int = 0,
    categorical_na_strategy: str = 'constant',
    categorical_fill_value: str = 'unknown',
    high_card_na_strategy: str = 'constant',
    high_card_fill_value: str = 'unknown',
    save_artifacts: bool = True,
    artifact_prefix: str = '',
    multi_input: bool = False,
    seed_worker: Optional[callable] = seed_worker,
    generator: Optional[torch.Generator] = g,
    log: bool = False,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline using text embeddings instead of TF-IDF.

    Args:
        X_train: The training input data as a pandas DataFrame.
        X_valid: The validation input data as a pandas DataFrame.
        X_test: The test input data as a pandas DataFrame.
        y_train: The training target values as a pandas Series.
        y_valid: The validation target values as a pandas Series.
        y_test: The test target values as a pandas Series.
        categorical_columns: A list of column names for low-cardinality categorical features to be one-hot encoded.
        high_cardinality_columns: A list of column names for high-cardinality categorical features to be target encoded.
        title_column: The name of the column containing the job title text.
        desc_column: The name of the column containing the job description text.
        embedding_model_name: The name of the SentenceTransformer model to use for text embeddings (e.g., 'all-MiniLM-L12-v2').
        batch_size: The batch size for the PyTorch DataLoaders.
        num_workers: The number of subprocesses to use for data loading.
        categorical_na_strategy: The imputation strategy for categorical columns ('constant' or 'most_frequent').
        categorical_fill_value: The value to use for filling missing values in categorical columns if `categorical_na_strategy` is 'constant'.
        high_card_na_strategy: The imputation strategy for high-cardinality columns ('constant' or 'most_frequent').
        high_card_fill_value: The value to use for filling missing values in high-cardinality columns if `high_card_na_strategy` is 'constant'.
        save_artifacts: Whether to save the fitted preprocessors (ColumnTransformer and StandardScaler).
        artifact_prefix: A string prefix for saved artifact filenames.
        multi_input: If True, creates a MultiInputDataset for models expecting separate inputs; otherwise, creates a standard SalaryDataset.
        seed_worker: Function to initialize each worker process (for reproducibility).
        generator: torch.Generator for DataLoader (for reproducibility).
        log: If True, indicates that y is already log-transformed, so skips target scaling and artifact saving.

    Returns:
        A dictionary containing:
            - 'train_loader', 'valid_loader', 'test_loader': PyTorch DataLoaders.
            - 'preprocessors': dict of fitted feature and target preprocessors.
            - 'feature_shape': shape of processed features.
            - 'processed_arrays': dict of the processed numpy arrays (X_train, y_train, etc.).
            - 'tabular_start_index': (Optional) index where tabular features start for multi-input models.
    """

    # Step 1: Build embedding pipelines
    title_embedding_pipeline = Pipeline([
        ('embedder', TextEmbedder(model_name=embedding_model_name))
    ])

    desc_embedding_pipeline = Pipeline([
        ('embedder', TextEmbedder(model_name=embedding_model_name))
    ])

    # Step 2: Setup categorical imputer
    if categorical_na_strategy == 'constant':
        cat_imputer = SimpleImputer(strategy='constant', fill_value=categorical_fill_value)
    else:
        cat_imputer = SimpleImputer(strategy='most_frequent')

    one_hot_pipeline = Pipeline([
        ('imputer', cat_imputer),  
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # Step 3: Setup high cardinality imputer
    if high_card_na_strategy == 'constant':
        high_card_imputer = SimpleImputer(strategy='constant', fill_value=high_card_fill_value)
    else:
        high_card_imputer = SimpleImputer(strategy='most_frequent')

    target_pipeline = Pipeline([
        ('imputer', high_card_imputer), 
        ('target_enc', TargetEncoder()), 
        ('scaler', StandardScaler())
    ])

    # Step 4: Combine into ColumnTransformer
    preprocessor_with_embeddings = ColumnTransformer([
        ('title_embeddings', title_embedding_pipeline, [title_column]),
        ('desc_embeddings', desc_embedding_pipeline, [desc_column]),
        ('onehot', one_hot_pipeline, categorical_columns),
        ('target_scaled', target_pipeline, high_cardinality_columns)
    ])

    # Step 5: Fit and transform features
    print(f"Using embedding model: {embedding_model_name}")
    print("Generating embeddings (this may take a while)...")
    X_train_processed = preprocessor_with_embeddings.fit_transform(X_train, y_train)
    X_valid_processed = preprocessor_with_embeddings.transform(X_valid)
    X_test_processed = preprocessor_with_embeddings.transform(X_test)
    print("Embeddings generated successfully!")

    # Step 6: Scale target (skip if log=True, as y is already log-transformed)
    if log:
        # y is already log-transformed, don't apply StandardScaler
        y_train_scaled = y_train.values
        y_valid_scaled = y_valid.values
        y_test_scaled = y_test.values
        target_scaler = None
        print("Log mode: Target is already log-transformed, skipping StandardScaler")
    else:
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # Step 7: Calculate tabular_start_index if multi_input
    tabular_start_index: Optional[int] = None
    if multi_input:
        # Get the fitted transformers to find their dimensions
        title_embedder = preprocessor_with_embeddings.named_transformers_['title_embeddings']['embedder']
        desc_embedder = preprocessor_with_embeddings.named_transformers_['desc_embeddings']['embedder']

        # Calculate the dimensions of each feature group
        embedding_dim: int = title_embedder.model.get_sentence_embedding_dimension()
        title_dim: int = embedding_dim
        desc_dim: int = embedding_dim

        # The tabular features start after title and description embeddings
        tabular_start_index = title_dim + desc_dim

        print(f"Embedding dimension: {embedding_dim}")
        print(f"Tabular start index: {tabular_start_index}")

    # Step 8: Create datasets
    if multi_input:
        if tabular_start_index is None:
            raise ValueError("Multi-input mode requires tabular_start_index, but it was not calculated.")
        train_dataset = MultiInputDataset(X_train_processed, y_train_scaled, tabular_start_index)
        valid_dataset = MultiInputDataset(X_valid_processed, y_valid_scaled, tabular_start_index)
        test_dataset = MultiInputDataset(X_test_processed, y_test_scaled, tabular_start_index)
    else:
        train_dataset = SalaryDataset(X_train_processed, y_train_scaled)
        valid_dataset = SalaryDataset(X_valid_processed, y_valid_scaled)
        test_dataset = SalaryDataset(X_test_processed, y_test_scaled)

    # Step 9: Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)

    # Step 10: Save artifacts (skip if log=True)
    if save_artifacts and not log:
        print("Saving preprocessor artifacts...")
        os.makedirs(PREPROCESSORS_DIR, exist_ok=True)

        preprocessor_path = PREPROCESSORS_DIR + f'{artifact_prefix}preprocessor_with_embeddings.pkl'
        scaler_path = PREPROCESSORS_DIR + f'{artifact_prefix}target_scaler.pkl'

        joblib.dump(preprocessor_with_embeddings, preprocessor_path)
        joblib.dump(target_scaler, scaler_path)
    elif log:
        print("Log mode: Skipping artifact saving")

    print(f"\nFinal feature shape: {X_train_processed.shape}")
    print(f"Categorical NA strategy: {categorical_na_strategy}")
    print(f"High cardinality NA strategy: {high_card_na_strategy}")
    print(f"Multi-input mode: {multi_input}")
    print(f"Log mode: {log}")

    result: Dict[str, Any] = {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'preprocessors': {
            'feature_prep': preprocessor_with_embeddings,
            'target_scaler': target_scaler
        },
        'feature_shape': X_train_processed.shape,
        'processed_arrays': {
            'X_train': X_train_processed,
            'X_valid': X_valid_processed,
            'X_test': X_test_processed,
            'y_train': y_train_scaled,
            'y_valid': y_valid_scaled,
            'y_test': y_test_scaled
        }
    }

    if multi_input and tabular_start_index is not None:
        result['tabular_start_index'] = tabular_start_index

    return result


# ### Function to preprocess data, create datasets and dataloaders for TF-IDF approach

# In[22]:


def preprocess_salary_data_with_tfidf(
    X_train: pd.DataFrame, 
    X_valid: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series, 
    y_valid: pd.Series, 
    y_test: pd.Series,
    categorical_columns: List[str] = ['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns: List[str] = ['Company', 'LocationNormalized', 'SourceName'],
    title_max_features: int = 50,
    desc_max_features: int = 800,
    title_use_svd: bool = False,
    title_n_components: int = 10,
    desc_use_svd: bool = False,
    desc_n_components: int = 50,
    title_stop_words: Optional[Union[str, List[str]]] = None,
    desc_stop_words: Optional[Union[str, List[str]]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    categorical_na_strategy: str = 'constant',
    categorical_fill_value: str = 'unknown',
    high_card_na_strategy: str = 'constant',
    high_card_fill_value: str = 'unknown',
    save_artifacts: bool = True,
    artifact_prefix: str = '',
    multi_input: bool = False,
    seed_worker: Optional[callable] = seed_worker,
    generator: Optional[torch.Generator] = g,
    log: bool = False,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for salary prediction data with TF-IDF.

    Args:
        X_train, X_valid, X_test: Input dataframes
        y_train, y_valid, y_test: Target values
        categorical_columns: List of low-cardinality categorical columns
        high_cardinality_columns: List of high-cardinality columns for target encoding
        title_max_features: Max TF-IDF features for title column
        desc_max_features: Max TF-IDF features for description column
        title_use_svd: Whether to apply SVD to title TF-IDF features
        title_n_components: Number of SVD components for title (if use_svd=True)
        desc_use_svd: Whether to apply SVD to description TF-IDF features
        desc_n_components: Number of SVD components for description (if use_svd=True)
        title_stop_words: Stop words for title vectorizer ('english', None, or list)
        desc_stop_words: Stop words for description vectorizer ('english', None, or list)
        batch_size: DataLoader batch size
        num_workers: DataLoader num_workers
        categorical_na_strategy: 'constant' or 'most_frequent' for categorical columns
        categorical_fill_value: Fill value when categorical_na_strategy='constant'
        high_card_na_strategy: 'constant' or 'most_frequent' for high cardinality columns
        high_card_fill_value: Fill value when high_card_na_strategy='constant'
        save_artifacts: Whether to save preprocessors
        artifact_prefix: Prefix for saved artifact filenames
        multi_input: If True, create MultiInputDataset; if False, create SalaryDataset
        seed_worker: Function to initialize each worker process (for reproducibility)
        generator: torch.Generator for DataLoader (for reproducibility)
        log: If True, indicates that y is already log-transformed, so skips target scaling and artifact saving.

    Returns:
        dict containing:
            - train_loader, valid_loader, test_loader: DataLoaders
            - preprocessors: dict of fitted preprocessors
            - feature_shape: shape of processed features
            - tabular_start_index: (only if multi_input=True) index where tabular features start
    """

    # Step 1: Clean text
    shared_text_prep = MinimalTextPreprocessor()
    X_train_clean = shared_text_prep.fit_transform(X_train)
    X_valid_clean = shared_text_prep.transform(X_valid)
    X_test_clean = shared_text_prep.transform(X_test)

    # Step 2: Build feature extraction pipelines
    title_pipeline = make_pipeline(
        TfidfTransformer(
            text_column='Title', 
            max_features=title_max_features,
            use_svd=title_use_svd,
            n_components=title_n_components,
            stop_words=title_stop_words
        )
    )

    desc_pipeline = make_pipeline(
        TfidfTransformer(
            text_column='FullDescription', 
            max_features=desc_max_features,
            use_svd=desc_use_svd,
            n_components=desc_n_components,
            stop_words=desc_stop_words
        )
    )

    # Categorical imputer setup
    if categorical_na_strategy == 'constant':
        cat_imputer = SimpleImputer(strategy='constant', fill_value=categorical_fill_value)
    else:
        cat_imputer = SimpleImputer(strategy='most_frequent')

    one_hot_pipeline = Pipeline([
        ('imputer', cat_imputer),  
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # High cardinality imputer setup
    if high_card_na_strategy == 'constant':
        high_card_imputer = SimpleImputer(strategy='constant', fill_value=high_card_fill_value)
    else:
        high_card_imputer = SimpleImputer(strategy='most_frequent')

    target_pipeline = Pipeline([
        ('imputer', high_card_imputer), 
        ('target_enc', TargetEncoder()),
        ('scaler', StandardScaler())
    ])

    # Step 3: Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('title_tfidf', title_pipeline, ['Title']),
        ('desc_tfidf', desc_pipeline, ['FullDescription']),
        ('onehot', one_hot_pipeline, categorical_columns),
        ('target_scaled', target_pipeline, high_cardinality_columns)
    ])

    # Step 4: Fit and transform features
    X_train_processed = preprocessor.fit_transform(X_train_clean, y_train)
    X_valid_processed = preprocessor.transform(X_valid_clean)
    X_test_processed = preprocessor.transform(X_test_clean)

    # Step 5: Scale target (skip if log=True, as y is already log-transformed)
    if log:
        # y is already log-transformed, don't apply StandardScaler
        y_train_scaled = y_train.values
        y_valid_scaled = y_valid.values
        y_test_scaled = y_test.values
        target_scaler = None
        print("Log mode: Target is already log-transformed, skipping StandardScaler")
    else:
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # Step 6: Calculate tabular_start_index if multi_input
    tabular_start_index = None
    if multi_input:
        # Calculate dimensions based on whether SVD was used
        if title_use_svd:
            title_dim = title_n_components
        else:
            title_dim = title_max_features

        if desc_use_svd:
            desc_dim = desc_n_components
        else:
            desc_dim = desc_max_features

        # The tabular features start after title and description TF-IDF features
        tabular_start_index = title_dim + desc_dim

        print(f"Title TF-IDF dimension: {title_dim}")
        print(f"Description TF-IDF dimension: {desc_dim}")
        print(f"Tabular start index: {tabular_start_index}")

    # Step 7: Create datasets
    if multi_input:
        train_dataset = MultiInputDataset(X_train_processed, y_train_scaled, tabular_start_index)
        valid_dataset = MultiInputDataset(X_valid_processed, y_valid_scaled, tabular_start_index)
        test_dataset = MultiInputDataset(X_test_processed, y_test_scaled, tabular_start_index)
    else:
        train_dataset = SalaryDataset(X_train_processed, y_train_scaled)
        valid_dataset = SalaryDataset(X_valid_processed, y_valid_scaled)
        test_dataset = SalaryDataset(X_test_processed, y_test_scaled)

    # Step 8: Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)

    # Step 9: Save artifacts (skip if log=True)
    if save_artifacts and not log:
        joblib.dump(shared_text_prep, PREPROCESSORS_DIR + f'{artifact_prefix}text_preprocessor.pkl')
        joblib.dump(preprocessor, PREPROCESSORS_DIR + f'{artifact_prefix}preprocessor.pkl')
        joblib.dump(target_scaler, PREPROCESSORS_DIR + f'{artifact_prefix}target_scaler.pkl')
    elif log:
        print("Log mode: Skipping artifact saving")

    # Print configuration info
    print(f"Final feature shape: {X_train_processed.shape}")
    print(f"Categorical NA strategy: {categorical_na_strategy}")
    print(f"High cardinality NA strategy: {high_card_na_strategy}")
    print(f"Title: max_features={title_max_features}, use_svd={title_use_svd}" + 
          (f", n_components={title_n_components}" if title_use_svd else ""))
    print(f"Description: max_features={desc_max_features}, use_svd={desc_use_svd}" + 
          (f", n_components={desc_n_components}" if desc_use_svd else ""))
    print(f"Multi-input mode: {multi_input}")
    print(f"Log mode: {log}")

    result = {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'preprocessors': {
            'text_prep': shared_text_prep,
            'feature_prep': preprocessor,
            'target_scaler': target_scaler
        },
        'feature_shape': X_train_processed.shape
    }

    if multi_input:
        result['tabular_start_index'] = tabular_start_index

    return result


# ### Function to train single input model

# In[23]:


def train_single_input_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    target_scaler = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    n_epochs: int = 25,
    lr: float = 0.001,
    loss_fn: str = 'mse',
    optimizer_fn: str = 'adam',
    patience: int = 4,
    delta: float = 0.001,
    early_stopping = None,
    use_lr_scheduler: bool = False,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    seed: Optional[int] = RANDOM_SEED,
    log: bool = False,
) -> Tuple[nn.Module, dict]:
    """
    Train a PyTorch regression model with early stopping.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        target_scaler: Fitted scaler for inverse transforming predictions (None if log=True)
        device: torch device (cuda or cpu)
        n_epochs: Maximum number of epochs
        lr: Learning rate
        loss_fn: Loss function ('mse' for Mean Squared Error, 'mae' for Mean Absolute Error, default: 'mse')
        optimizer_fn: Optimizer function ('adam' or 'sgd', default: 'adam')
        patience: Early stopping patience
        delta: Minimum change to qualify as improvement
        early_stopping: EarlyStopping object (if None, creates one)
        use_lr_scheduler: Whether to use ReduceLROnPlateau scheduler
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor to reduce learning rate by
        seed: Random seed for reproducibility (None to skip seeding)
        log: If True, indicates that y is log-transformed (e.g., using log1p); real metrics computed using expm1

    Returns:
        Tuple of (trained_model, history_dict, elapsed_time)
    """
    # Set seed for reproducibility
    if seed is not None:
        set_seed(seed)

    # Move model to device
    model = model.to(device)

    # Setup loss function
    if loss_fn == 'mse':
        loss_function = nn.MSELoss()
    elif loss_fn == 'mae':
        loss_function = nn.L1Loss()
    else:
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Setup optimizer
    if optimizer_fn == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_fn == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")

    # Initialize learning rate scheduler if requested
    scheduler = None
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience
        )

    # Initialize early stopping if not provided
    if early_stopping is None:
        early_stopping = EarlyStopping(
            patience=patience, 
            delta=delta, 
            verbose=True, 
            restore_best_weights=True
        )

    # Lists to store metrics based on loss function
    if loss_fn == 'mse':
        history = {
            'train_loss_scaled': [],
            'valid_loss_scaled': [],
            'train_mse_real': [],
            'valid_mse_real': [],
            'train_rmse': [],
            'valid_rmse': []
        }
    else:  # mae
        history = {
            'train_loss_scaled': [],
            'valid_loss_scaled': [],
            'train_mae_real': [],
            'valid_mae_real': []
        }

    start_time = time.time()
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss_scaled = 0.0
        train_loss_real = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(X_batch).squeeze()

            # Compute loss
            loss = loss_function(predictions, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss_scaled += loss.item() * X_batch.size(0)

            # Compute metric in real scale
            if log:
                # For log1p-transformed targets: use expm1 to get back to original scale
                # expm1(x) = exp(x) - 1, which is the inverse of log1p
                predictions_real = np.expm1(predictions.detach().cpu().numpy())
                y_batch_real = np.expm1(y_batch.detach().cpu().numpy())
            else:
                # For scaled targets: inverse transform using scaler
                if target_scaler is None:
                    raise ValueError("target_scaler must be provided when log=False")
                predictions_real = target_scaler.inverse_transform(
                    predictions.detach().cpu().numpy().reshape(-1, 1)
                ).ravel()
                y_batch_real = target_scaler.inverse_transform(
                    y_batch.detach().cpu().numpy().reshape(-1, 1)
                ).ravel()

            if loss_fn == 'mse':
                metric_real = np.mean((predictions_real - y_batch_real) ** 2)
            else:  # mae
                metric_real = np.mean(np.abs(predictions_real - y_batch_real))

            train_loss_real += metric_real * X_batch.size(0)

        # Calculate average training losses
        train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
        train_loss_real_avg = train_loss_real / len(train_loader.dataset)

        if loss_fn == 'mse':
            train_rmse = np.sqrt(train_loss_real_avg)

        # Validation phase
        model.eval()
        valid_loss_scaled = 0.0
        valid_loss_real = 0.0

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Forward pass
                predictions = model(X_batch).squeeze()

                # Use specified loss function for validation loss
                loss = loss_function(predictions, y_batch)
                valid_loss_scaled += loss.item() * X_batch.size(0)

                # Compute metric in real scale
                if log:
                    # For log1p-transformed targets: use expm1 to get back to original scale
                    # expm1(x) = exp(x) - 1, which is the inverse of log1p
                    predictions_real = np.expm1(predictions.detach().cpu().numpy())
                    y_batch_real = np.expm1(y_batch.detach().cpu().numpy())
                else:
                    # For scaled targets: inverse transform using scaler
                    if target_scaler is None:
                        raise ValueError("target_scaler must be provided when log=False")
                    predictions_real = target_scaler.inverse_transform(
                        predictions.detach().cpu().numpy().reshape(-1, 1)
                    ).ravel()
                    y_batch_real = target_scaler.inverse_transform(
                        y_batch.detach().cpu().numpy().reshape(-1, 1)
                    ).ravel()

                if loss_fn == 'mse':
                    metric_real = np.mean((predictions_real - y_batch_real) ** 2)
                else:  # mae
                    metric_real = np.mean(np.abs(predictions_real - y_batch_real))

                valid_loss_real += metric_real * X_batch.size(0)

        # Calculate average validation losses
        valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
        valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)

        if loss_fn == 'mse':
            valid_rmse = np.sqrt(valid_loss_real_avg)

        # Store metrics
        history['train_loss_scaled'].append(train_loss_scaled_avg)
        history['valid_loss_scaled'].append(valid_loss_scaled_avg)

        if loss_fn == 'mse':
            history['train_mse_real'].append(train_loss_real_avg)
            history['valid_mse_real'].append(valid_loss_real_avg)
            history['train_rmse'].append(train_rmse)
            history['valid_rmse'].append(valid_rmse)
        else:  # mae
            history['train_mae_real'].append(train_loss_real_avg)
            history['valid_mae_real'].append(valid_loss_real_avg)

        # Print epoch results
        print(f'Epoch {epoch+1}/{n_epochs}:')
        if loss_fn == 'mse':
            print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {train_rmse:.2f}')
            print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {valid_rmse:.2f}')
        else:  # mae
            print(f'  Train - MAE: {train_loss_scaled_avg:.4f}, Real MAE: {train_loss_real_avg:.2f}')
            print(f'  Valid - MAE: {valid_loss_scaled_avg:.4f}, Real MAE: {valid_loss_real_avg:.2f}')

        # Step the learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step(valid_loss_scaled_avg)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Learning Rate: {current_lr:.6f}')

        # Check early stopping
        early_stopping.check_early_stop(valid_loss_scaled_avg, model)
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch+1}")
            break

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes.")

    return model, history, elapsed_time


# ### Function to train multi-input model

# In[24]:


def train_multi_input_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    target_scaler = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    n_epochs: int = 25,
    lr: float = 0.001,
    loss_fn: str = 'mse',
    optimizer_fn: str = 'adam',
    patience: int = 3,
    delta: float = 0.001,
    early_stopping = None,
    use_lr_scheduler: bool = False,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    seed: Optional[int] = RANDOM_SEED,
    log: bool = False,
) -> Tuple[nn.Module, dict]:
    """
    Train a PyTorch regression model with early stopping.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        target_scaler: Fitted scaler for inverse transforming predictions (None if log=True)
        device: torch device (cuda or cpu)
        n_epochs: Maximum number of epochs
        lr: Learning rate
        loss_fn: Loss function ('mse' for Mean Squared Error, 'mae' for Mean Absolute Error, default: 'mse')
        optimizer_fn: Optimizer function ('adam' or 'sgd', default: 'adam')
        patience: Early stopping patience
        delta: Minimum change to qualify as improvement
        early_stopping: EarlyStopping object (if None, creates one)
        use_lr_scheduler: Whether to use ReduceLROnPlateau scheduler
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor to reduce learning rate by
        seed: Random seed for reproducibility (None to skip seeding)
        log: If True, indicates that y is log-transformed (e.g., using log1p); real metrics computed using expm1

    Returns:
        Tuple of (trained_model, history_dict)
    """
    # Set seed for reproducibility
    if seed is not None:
        set_seed(seed)

    # Move model to device
    model = model.to(device)

    # Setup loss function
    if loss_fn == 'mse':
        loss_function = nn.MSELoss()
    elif loss_fn == 'mae':
        loss_function = nn.L1Loss()
    else:
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Setup optimizer
    if optimizer_fn == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_fn == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")

    # Initialize learning rate scheduler if requested
    scheduler = None
    if use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience
        )

    # Initialize early stopping if not provided
    if early_stopping is None:
        early_stopping = EarlyStopping(
            patience=patience, 
            delta=delta, 
            verbose=True, 
            restore_best_weights=True
        )

    # Lists to store metrics based on loss function
    if loss_fn == 'mse':
        history = {
            'train_loss_scaled': [],
            'valid_loss_scaled': [],
            'train_mse_real': [],
            'valid_mse_real': [],
            'train_rmse': [],
            'valid_rmse': []
        }
    else:  # mae
        history = {
            'train_loss_scaled': [],
            'valid_loss_scaled': [],
            'train_mae_real': [],
            'valid_mae_real': []
        }

    start_time = time.time()
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss_scaled = 0.0
        train_loss_real = 0.0

        # --- Multi-Input Adaptation ---
        for embeddings_batch, tabular_batch, y_batch in train_loader:
            embeddings_batch, tabular_batch, y_batch = (
                embeddings_batch.to(device),
                tabular_batch.to(device),
                y_batch.to(device)
            )

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with two inputs
            predictions = model(embeddings_batch, tabular_batch).squeeze()

            # Compute loss
            loss = loss_function(predictions, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss_scaled += loss.item() * embeddings_batch.size(0)

            # Compute metric in real scale
            if log:
                # For log1p-transformed targets: use expm1 to get back to original scale
                # expm1(x) = exp(x) - 1, which is the inverse of log1p
                predictions_real = np.expm1(predictions.detach().cpu().numpy())
                y_batch_real = np.expm1(y_batch.detach().cpu().numpy())
            else:
                # For scaled targets: inverse transform using scaler
                if target_scaler is None:
                    raise ValueError("target_scaler must be provided when log=False")
                predictions_real = target_scaler.inverse_transform(
                    predictions.detach().cpu().numpy().reshape(-1, 1)
                ).ravel()
                y_batch_real = target_scaler.inverse_transform(
                    y_batch.detach().cpu().numpy().reshape(-1, 1)
                ).ravel()

            if loss_fn == 'mse':
                metric_real = np.mean((predictions_real - y_batch_real) ** 2)
            else:  # mae
                metric_real = np.mean(np.abs(predictions_real - y_batch_real))

            train_loss_real += metric_real * embeddings_batch.size(0)

        # Calculate average training losses
        train_loss_scaled_avg = train_loss_scaled / len(train_loader.dataset)
        train_loss_real_avg = train_loss_real / len(train_loader.dataset)

        if loss_fn == 'mse':
            train_rmse = np.sqrt(train_loss_real_avg)

        # Validation phase
        model.eval()
        valid_loss_scaled = 0.0
        valid_loss_real = 0.0

        with torch.no_grad():
            # --- Multi-Input Adaptation ---
            for embeddings_batch, tabular_batch, y_batch in valid_loader:
                embeddings_batch, tabular_batch, y_batch = (
                    embeddings_batch.to(device),
                    tabular_batch.to(device),
                    y_batch.to(device)
                )

                # Forward pass with two inputs
                predictions = model(embeddings_batch, tabular_batch).squeeze()

                # Use specified loss function for validation loss
                loss = loss_function(predictions, y_batch)
                valid_loss_scaled += loss.item() * embeddings_batch.size(0)

                # Compute metric in real scale
                if log:
                    # For log1p-transformed targets: use expm1 to get back to original scale
                    # expm1(x) = exp(x) - 1, which is the inverse of log1p
                    predictions_real = np.expm1(predictions.detach().cpu().numpy())
                    y_batch_real = np.expm1(y_batch.detach().cpu().numpy())
                else:
                    # For scaled targets: inverse transform using scaler
                    if target_scaler is None:
                        raise ValueError("target_scaler must be provided when log=False")
                    predictions_real = target_scaler.inverse_transform(
                        predictions.detach().cpu().numpy().reshape(-1, 1)
                    ).ravel()
                    y_batch_real = target_scaler.inverse_transform(
                        y_batch.detach().cpu().numpy().reshape(-1, 1)
                    ).ravel()

                if loss_fn == 'mse':
                    metric_real = np.mean((predictions_real - y_batch_real) ** 2)
                else:  # mae
                    metric_real = np.mean(np.abs(predictions_real - y_batch_real))

                valid_loss_real += metric_real * embeddings_batch.size(0)

        # Calculate average validation losses
        valid_loss_scaled_avg = valid_loss_scaled / len(valid_loader.dataset)
        valid_loss_real_avg = valid_loss_real / len(valid_loader.dataset)

        if loss_fn == 'mse':
            valid_rmse = np.sqrt(valid_loss_real_avg)

        # Store metrics
        history['train_loss_scaled'].append(train_loss_scaled_avg)
        history['valid_loss_scaled'].append(valid_loss_scaled_avg)

        if loss_fn == 'mse':
            history['train_mse_real'].append(train_loss_real_avg)
            history['valid_mse_real'].append(valid_loss_real_avg)
            history['train_rmse'].append(train_rmse)
            history['valid_rmse'].append(valid_rmse)
        else:  # mae
            history['train_mae_real'].append(train_loss_real_avg)
            history['valid_mae_real'].append(valid_loss_real_avg)

        # Print epoch results
        print(f'Epoch {epoch+1}/{n_epochs}:')
        if loss_fn == 'mse':
            print(f'  Train - MSE: {train_loss_scaled_avg:.4f}, Real MSE: {train_loss_real_avg:.2f}, Real RMSE: {train_rmse:.2f}')
            print(f'  Valid - MSE: {valid_loss_scaled_avg:.4f}, Real MSE: {valid_loss_real_avg:.2f}, Real RMSE: {valid_rmse:.2f}')
        else:  # mae
            print(f'  Train - MAE: {train_loss_scaled_avg:.4f}, Real MAE: {train_loss_real_avg:.2f}')
            print(f'  Valid - MAE: {valid_loss_scaled_avg:.4f}, Real MAE: {valid_loss_real_avg:.2f}')

        # Step the learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step(valid_loss_scaled_avg)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Learning Rate: {current_lr:.6f}')

        # Check early stopping
        early_stopping.check_early_stop(valid_loss_scaled_avg, model)
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch+1}")
            break

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes.")

    return model, history, elapsed_time


# ### Function to evaluate single input model on test set and store results in dictionary

# In[25]:


def evaluate_single_input_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    target_scaler = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    loss_fn_str: str = 'mse',
    model_name: str = None,
    results_dict: Dict = None,
    log: bool = False,
) -> Dict:
    """
    Evaluates a single-input PyTorch regression model and stores results in a dictionary.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test data.
        target_scaler: The fitted scaler for inverse transforming predictions (None if log=True).
        device: torch device (cuda or cpu).
        loss_fn_str: The metric to calculate ('mse' or 'mae').
        model_name: Name/identifier for the model (optional).
        results_dict: Dictionary to store results (optional, will update in-place).
        log: If True, indicates that y is log-transformed (e.g., using log1p); real metrics computed using expm1.

    Returns:
        A dictionary containing the calculated metrics.
    """
    model.eval()

    # Select the appropriate loss function
    if loss_fn_str == 'mse':
        loss_function = nn.MSELoss()
    elif loss_fn_str == 'mae':
        loss_function = nn.L1Loss()
    else:
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Metrics for evaluation
    test_loss_scaled = 0.0
    all_predictions_real = []
    all_targets_real = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch).squeeze()

            # Accumulate loss in scaled domain
            loss = loss_function(predictions, y_batch)
            test_loss_scaled += loss.item() * X_batch.size(0)

            # Inverse transform to real scale
            if log:
                # For log1p-transformed targets: use expm1 to get back to original scale
                # expm1(x) = exp(x) - 1, which is the inverse of log1p
                predictions_real = np.expm1(predictions.cpu().numpy())
                y_batch_real = np.expm1(y_batch.cpu().numpy())
            else:
                # For scaled targets: inverse transform using scaler
                if target_scaler is None:
                    raise ValueError("target_scaler must be provided when log=False")
                predictions_real = target_scaler.inverse_transform(
                    predictions.cpu().numpy().reshape(-1, 1)
                ).ravel()
                y_batch_real = target_scaler.inverse_transform(
                    y_batch.cpu().numpy().reshape(-1, 1)
                ).ravel()

            # Store predictions and targets for real-scale metrics
            all_predictions_real.extend(predictions_real)
            all_targets_real.extend(y_batch_real)

    # Average loss over the entire dataset
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)

    # Calculate real-scale metrics over the entire dataset
    all_predictions_real = np.array(all_predictions_real)
    all_targets_real = np.array(all_targets_real)

    metrics = {}
    if loss_fn_str == 'mse':
        mse_real = np.mean((all_predictions_real - all_targets_real) ** 2)
        metrics['Test MSE (scaled)'] = test_loss_scaled_avg
        metrics['Test MSE (real)'] = mse_real
        metrics['Test RMSE (real)'] = np.sqrt(mse_real)
    elif loss_fn_str == 'mae':
        mae_real = np.mean(np.abs(all_predictions_real - all_targets_real))
        metrics['Test MAE (scaled)'] = test_loss_scaled_avg
        metrics['Test MAE (real)'] = mae_real

    # Store in results_dict if provided
    if results_dict is not None and model_name is not None:
        results_dict[model_name] = metrics

    return metrics


# ### Function to evaluate multi-input model on test set and store results in dictionary

# In[26]:


def evaluate_multi_input_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_scaler = None,
    loss_fn_str: str = 'mse',
    model_name: str = None,
    results_dict: Dict = None,
    log: bool = False,
) -> Dict:
    """
    Evaluates a multi-input PyTorch regression model and stores results in a dictionary.

    Args:
        model: The trained PyTorch model (expects multi-input).
        test_loader: DataLoader for the test data (MultiInputDataset).
        target_scaler: The fitted scaler for inverse transforming predictions.
        device: torch device (cuda or cpu).
        loss_fn_str: The metric to calculate ('mse' or 'mae').
        model_name: Name/identifier for the model (optional).
        results_dict: Dictionary to store results (optional, will update in-place).
        log: If True, indicates that y is log-transformed (e.g., using log1p); real metrics computed using expm1.

    Returns:
        A dictionary containing the calculated metrics.
    """
    model.eval()

    # Select the appropriate loss function
    if loss_fn_str == 'mse':
        loss_function = nn.MSELoss()
    elif loss_fn_str == 'mae':
        loss_function = nn.L1Loss()
    else:
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Metrics for evaluation
    test_loss_scaled = 0.0
    all_predictions_real = []
    all_targets_real = []

    with torch.no_grad():
        for embeddings_batch, tabular_batch, y_batch in test_loader:
            embeddings_batch = embeddings_batch.to(device)
            tabular_batch = tabular_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass (multi-input)
            predictions = model(embeddings_batch, tabular_batch).squeeze()

            # Accumulate loss in scaled domain
            loss = loss_function(predictions, y_batch)
            test_loss_scaled += loss.item() * embeddings_batch.size(0)

            # Inverse transform to real scale
            if log:
                # For log1p-transformed targets: use expm1 to get back to original scale
                # expm1(x) = exp(x) - 1, which is the inverse of log1p
                predictions_real = np.expm1(predictions.cpu().numpy())
                y_batch_real = np.expm1(y_batch.cpu().numpy())
            else:
                # For scaled targets: inverse transform using scaler
                if target_scaler is None:
                    raise ValueError("target_scaler must be provided when log=False")
                predictions_real = target_scaler.inverse_transform(
                    predictions.cpu().numpy().reshape(-1, 1)
                ).ravel()
                y_batch_real = target_scaler.inverse_transform(
                    y_batch.cpu().numpy().reshape(-1, 1)
                ).ravel()

            # Store predictions and targets for real-scale metrics
            all_predictions_real.extend(predictions_real)
            all_targets_real.extend(y_batch_real)

    # Average loss over the entire dataset
    test_loss_scaled_avg = test_loss_scaled / len(test_loader.dataset)

    # Calculate real-scale metrics over the entire dataset
    all_predictions_real = np.array(all_predictions_real)
    all_targets_real = np.array(all_targets_real)

    metrics = {}
    if loss_fn_str == 'mse':
        mse_real = np.mean((all_predictions_real - all_targets_real) ** 2)
        metrics['Test MSE (scaled)'] = test_loss_scaled_avg
        metrics['Test MSE (real)'] = mse_real
        metrics['Test RMSE (real)'] = np.sqrt(mse_real)
    elif loss_fn_str == 'mae':
        mae_real = np.mean(np.abs(all_predictions_real - all_targets_real))
        metrics['Test MAE (scaled)'] = test_loss_scaled_avg
        metrics['Test MAE (real)'] = mae_real

    # Store in results_dict if provided
    if results_dict is not None and model_name is not None:
        results_dict[model_name] = metrics

    return metrics


# ### Function to plot training and validation curves from dict for given loss function

# In[27]:


def plot_losses_curves(history_dict, loss_fn='mse'):
    """
    Plot training and validation losses from a history dictionary.
    Left plot shows scaled losses, right plot shows real-scale losses.

    Args:
        history_dict: Dictionary containing loss histories
        loss_fn: 'mse' or 'mae' - determines which losses to plot
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if loss_fn == 'mse':
        # Plot 1: Scaled MSE (left)
        epochs = range(1, len(history_dict['train_loss_scaled']) + 1)
        axes[0].plot(epochs, history_dict['train_loss_scaled'], color='blue', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history_dict['valid_loss_scaled'], color='orange', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss (Scaled MSE)', fontsize=11)
        axes[0].set_title('Scaled MSE Loss', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Real MSE (right)
        axes[1].plot(epochs, history_dict['train_mse_real'], color='blue', label='Train Loss', linewidth=2)
        axes[1].plot(epochs, history_dict['valid_mse_real'], color='orange', label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss (Real MSE)', fontsize=11)
        axes[1].set_title('Real Scale MSE Loss', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    else:  # mae
        # Plot 1: Scaled MAE (left)
        epochs = range(1, len(history_dict['train_loss_scaled']) + 1)
        axes[0].plot(epochs, history_dict['train_loss_scaled'], color='blue', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history_dict['valid_loss_scaled'], color='orange', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss (Scaled MAE)', fontsize=11)
        axes[0].set_title('Scaled MAE Loss', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Real MAE (right)
        axes[1].plot(epochs, history_dict['train_mae_real'], color='blue', label='Train Loss', linewidth=2)
        axes[1].plot(epochs, history_dict['valid_mae_real'], color='orange', label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss (Real MAE)', fontsize=11)
        axes[1].set_title('Real Scale MAE Loss', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ### Complete process for TF-IDF approach with single input model

# In[174]:


X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data()


# In[175]:


preprocessing_results = preprocess_salary_data_with_tfidf(
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    categorical_columns=['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns=['Company', 'LocationNormalized', 'SourceName'],
    title_max_features=50,
    desc_max_features=800,
    batch_size=64,
    num_workers=0,
    categorical_na_strategy='constant',  # 'constant' or 'most_frequent'
    categorical_fill_value='unknown',    # used when strategy='constant'
    high_card_na_strategy='constant',    # 'constant' or 'most_frequent'
    high_card_fill_value='unknown',      # used when strategy='constant'
    save_artifacts=True,
    artifact_prefix=''
)
train_loader = preprocessing_results['train_loader']
valid_loader = preprocessing_results['valid_loader']
test_loader = preprocessing_results['test_loader']


# In[176]:


model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1]).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=False,
    seed=RANDOM_SEED
)


# In[177]:


plot_losses_curves(history, loss_fn='mse')


# #### **Best validation scaled MSE: 0.282282**

# In[178]:


print_model_parameters_summary(model)


# In[179]:


model_name = 'tfidf_800_50_int_unk_bs64_adam_lrs_no_hid256_dr30'
torch.save(trained_model.state_dict(), MODELS_DIR + f'{model_name}.pth')


# In[180]:


evaluation_metrics = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# ### Complete process for pretrained embeddings approach with single input model

# In[181]:


X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data()


# In[182]:


preprocessing_results = preprocess_salary_data_with_embeddings(
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    categorical_columns=['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns=['Company', 'LocationNormalized', 'SourceName'],
    title_column='Title',
    desc_column='FullDescription',
    embedding_model_name='all-MiniLM-L12-v2',
    batch_size=64,
    num_workers=0,
    categorical_na_strategy='constant',
    categorical_fill_value='unknown',
    high_card_na_strategy='constant',
    high_card_fill_value='unknown',
    save_artifacts=True,
    artifact_prefix='',
    multi_input=False
)
train_loader = preprocessing_results['train_loader']
valid_loader = preprocessing_results['valid_loader']
test_loader = preprocessing_results['test_loader']


# In[ ]:


model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1]).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=False,
    seed=RANDOM_SEED
)


# In[184]:


plot_losses_curves(history, loss_fn='mse')


# #### **Best validation scaled MSE: 0.272884**

# In[185]:


print_model_parameters_summary(model)


# In[186]:


model_name = 'emb_int_unk_bs64_adam_lrs_no_hid256_dr30'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[187]:


results = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# ### Complete process for pretrained embeddings approach with multi-input model

# In[188]:


X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data()


# In[189]:


preprocessing_results = preprocess_salary_data_with_embeddings(
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    categorical_columns=['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns=['Company', 'LocationNormalized', 'SourceName'],
    title_column='Title',
    desc_column='FullDescription',
    embedding_model_name='all-MiniLM-L12-v2',
    batch_size=64,
    num_workers=0,
    categorical_na_strategy='constant',
    categorical_fill_value='unknown',
    high_card_na_strategy='constant',
    high_card_fill_value='unknown',
    save_artifacts=True,
    artifact_prefix='',
    multi_input=True
)
train_loader = preprocessing_results['train_loader']
valid_loader = preprocessing_results['valid_loader']
test_loader = preprocessing_results['test_loader']
embedding_dim = preprocessing_results['tabular_start_index']


# In[ ]:


model = MultiInputNN(embedding_dim=embedding_dim, tabular_dim=preprocessing_results['feature_shape'][1] - embedding_dim).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_multi_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=False,
    seed=RANDOM_SEED
)


# In[191]:


plot_losses_curves(history, loss_fn='mse')


# #### **Best validation scaled MSE: 0.272726**

# In[192]:


print_model_parameters_summary(model)


# In[193]:


model_name = 'emb_multi_unk_bs64_adam_lrs_no_hid256_dr30'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[194]:


results = evaluate_multi_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# ### Complete process for multi-input model with TF-IDF

# In[195]:


X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data()


# In[196]:


preprocessing_results = preprocess_salary_data_with_tfidf(
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    categorical_columns=['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns=['Company', 'LocationNormalized', 'SourceName'],
    title_max_features=50,
    desc_max_features=800,
    batch_size=64,
    num_workers=0,
    categorical_na_strategy='constant',  # 'constant' or 'most_frequent'
    categorical_fill_value='unknown',    # used when strategy='constant'
    high_card_na_strategy='constant',    # 'constant' or 'most_frequent'
    high_card_fill_value='unknown',      # used when strategy='constant'
    save_artifacts=True,
    artifact_prefix='',
    multi_input=True
)
train_loader = preprocessing_results['train_loader']
valid_loader = preprocessing_results['valid_loader']
test_loader = preprocessing_results['test_loader']
embedding_dim = preprocessing_results['tabular_start_index']


# In[ ]:


model = MultiInputNN(embedding_dim=embedding_dim, tabular_dim=preprocessing_results['feature_shape'][1] - embedding_dim).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_multi_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=False,
    seed=RANDOM_SEED
)


# In[198]:


plot_losses_curves(history, loss_fn='mse')


# #### **Best validation scaled MSE: 0.281541.**

# In[199]:


model_name = 'tfidf_800_50_multi_unk_bs64_adam_lrs_no_hid256_dr30'
torch.save(model.state_dict(), MODELS_DIR + model_name + '.pth')


# In[200]:


results = evaluate_multi_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# ### Results summary

# 
# We can see that pretrained embeddings give better results than TF-IDF approach. However, the difference is not very large and pretrained embeddings take longer to prepare data for training. Multi-input models do not perform better than single-input models in this case.
# 

# ### Tuning hyperparameters for single-input model with TF-IDF

# #### Filling missing values with most frequent value instead of 'Unknown'

# In[ ]:


X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data()

preprocessing_results = preprocess_salary_data_with_tfidf(
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    categorical_columns=['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns=['Company', 'LocationNormalized', 'SourceName'],
    title_max_features=50,
    desc_max_features=800,
    batch_size=64,
    num_workers=0,
    categorical_na_strategy='most_frequent',  # 'constant' or 'most_frequent'
    high_card_na_strategy='most_frequent',    # 'constant' or 'most_frequent'
    save_artifacts=False
)
train_loader = preprocessing_results['train_loader']
valid_loader = preprocessing_results['valid_loader']
test_loader = preprocessing_results['test_loader']

model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1]).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=False,
    seed=RANDOM_SEED
)

plot_losses_curves(history, loss_fn='mse')


# **Validation Scaled MSE result: 0.281541 - better result.** From now we will use this approach.

# In[202]:


print_model_parameters_summary(model)


# In[203]:


model_name = 'tfidf_800_50_int_mf_bs64_adam_lrs_no_hid256_dr30'
torch.save(trained_model.state_dict(), MODELS_DIR + f'{model_name}.pth')


# In[204]:


results = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# #### Using Simpler Model with one less layer and less neurons

# In[ ]:


model = SimpleRegressorWithNormalization(input_dim=preprocessing_results['feature_shape'][1], hidden_size=64).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=False,
    seed=RANDOM_SEED
)

plot_losses_curves(history, loss_fn='mse')


# **0.307010 - Worse result.** We will not use this approach in further experiments.

# In[206]:


print_model_parameters_summary(model)


# In[207]:


model_name = 'tfidf_800_50_srbn_unk_bs64_adam_lrs_no_hid64_dr30'
torch.save(trained_model.state_dict(), MODELS_DIR + f'{model_name}.pth')

results = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# #### More complex model but with 2 times fewer neurons in each layer

# In[208]:


model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1], hidden_size=128).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=False,
    seed=RANDOM_SEED
)

plot_losses_curves(history, loss_fn='mse')


# **0.292486- Worse result.** 

# In[209]:


print_model_parameters_summary(model)


# In[210]:


model_name = 'tfidf_800_50_int_unk_bs64_adam_lrs_no_hid128_dr30'
torch.save(trained_model.state_dict(), MODELS_DIR + f'{model_name}.pth')

results = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# #### Using learning rate scheduler

# In[ ]:


model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1]).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=True,
    scheduler_patience=0,
    scheduler_factor=0.5,
    seed=RANDOM_SEED
)

plot_losses_curves(history, loss_fn='mse')


# **0.276861 - better result - using patience=0, factor=0.5 for scheduler may help stabilize training**

# In[212]:


print_model_parameters_summary(model)


# In[213]:


model_name = 'tfidf_800_50_int_unk_bs64_adam_lrs_hid256_dr30'
torch.save(trained_model.state_dict(), MODELS_DIR + f'{model_name}.pth')

results = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# #### Using other optimizer - SGD with momentum=0.9

# In[ ]:


model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1]).to(device)
loss_fn = 'mse'
n_epochs = 50
optimizer_fn = 'sgd'
patience = 5
lr = 0.005

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=True,
    scheduler_patience=0,
    scheduler_factor=0.5,
    seed=RANDOM_SEED
)

plot_losses_curves(history, loss_fn='mse')


# **Adam seems to work better - it is faster to converge.**

# In[215]:


print_model_parameters_summary(model)


# In[216]:


model_name = 'tfidf_800_50_int_unk_bs64_sgd_lrs_no_hid256_dr30'
torch.save(trained_model.state_dict(), MODELS_DIR + f'{model_name}.pth')

results = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',
    model_name=model_name,
    results_dict=fulldata_best_models
)


# Ok, so the best model for single-input with TF-IDF is the one with filling missing values with most frequent value, IntegratedNN model, batch size of 64, Adam optimizer and learning rate scheduler.

# Now I will try to modify TfidfVectorizer parameters.

# #### Adding svd to reduce dimensionality of text features

# In[ ]:


preprocessing_results = preprocess_salary_data_with_tfidf(
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    categorical_columns=['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns=['Company', 'LocationNormalized', 'SourceName'],
    title_max_features=50,
    desc_max_features=800,
    desc_use_svd=True,
    desc_n_components=200,
    title_use_svd=True,
    title_n_components=10,
    batch_size=64,
    num_workers=0,
    categorical_na_strategy='most_frequent',  # 'constant' or 'most_frequent'
    high_card_na_strategy='most_frequent',    # 'constant' or 'most_frequent'
    save_artifacts=False
)
train_loader = preprocessing_results['train_loader']
valid_loader = preprocessing_results['valid_loader']
test_loader = preprocessing_results['test_loader']

model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1]).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=True,
    scheduler_patience=0,
    scheduler_factor=0.5,
    seed=RANDOM_SEED
)

plot_losses_curves(history, loss_fn='mse')


# **The validation Scaled MSE is 0.320322 - definitely worse result - although the training was much faster. We need to select between performance and training time.**

# In[218]:


print_model_parameters_summary(model)


# In[219]:


model_name = 'tfidf_scd_200_10_int_unk_bs64_adam_lrs_no_hid256_dr30'
torch.save(trained_model.state_dict(), MODELS_DIR + f'{model_name}.pth')
results = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',  
    model_name=model_name,
    results_dict=fulldata_best_models
)


# #### Using remove stop words for title and description in TfidfVectorizer

# In[ ]:


preprocessing_results = preprocess_salary_data_with_tfidf(
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    categorical_columns=['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns=['Company', 'LocationNormalized', 'SourceName'],
    title_max_features=50,
    desc_max_features=800,
    batch_size=64,
    num_workers=0,
    desc_stop_words='english', # using stop words
    title_stop_words='english', # using stop words
    categorical_na_strategy='most_frequent',  
    high_card_na_strategy='most_frequent',    
    save_artifacts=False
)
train_loader = preprocessing_results['train_loader']
valid_loader = preprocessing_results['valid_loader']
test_loader = preprocessing_results['test_loader']

model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1]).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    target_scaler=preprocessing_results['preprocessors']['target_scaler'],
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=True,
    scheduler_patience=0,
    scheduler_factor=0.5,
    seed=RANDOM_SEED
)

plot_losses_curves(history, loss_fn='mse')


# **Validation loss improved to 0.261083- better than previous best result - removing stop words seems to help**

# In[221]:


print_model_parameters_summary(model)


# In[222]:


model_name = 'tfidf_stopwords_800_50_int_unk_bs64_adam_lrs_no_hid256_dr30'
torch.save(trained_model.state_dict(), MODELS_DIR + f'{model_name}.pth')
results = evaluate_single_input_model(
    model,
    test_loader,
    preprocessing_results['preprocessors']['target_scaler'],
    device,
    loss_fn_str='mse',  
    model_name=model_name,
    results_dict=fulldata_best_models
)


# ## Compare results of best models on the test set

# In[223]:


fulldata_best_models


# In[224]:


def plot_top_models_comparison(
    results_dict: Dict = fulldata_best_models,
    top_n: int = 10,
    figsize: tuple = (16, 6),
):
    """
    Creates side-by-side bar plots comparing models by Test MSE (scaled) and Test RMSE (real).

    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        top_n: Number of top models to display (default: 10)
        figsize: Figure size as (width, height) tuple
        save_path: Optional path to save the figure
    """
    # Extract data from results dictionary
    model_names = list(results_dict.keys())
    mse_scaled = [results_dict[name]['Test MSE (scaled)'] for name in model_names]
    rmse_real = [float(results_dict[name]['Test RMSE (real)']) for name in model_names]

    # Create dataframe-like structure for sorting
    data = list(zip(model_names, mse_scaled, rmse_real))

    # Sort by MSE (scaled) and get top N
    data_sorted_mse = sorted(data, key=lambda x: x[1])[:top_n]
    top_names_mse = [x[0] for x in data_sorted_mse]
    top_mse_scaled = [x[1] for x in data_sorted_mse]
    top_mse_real = [results_dict[x]['Test MSE (real)'] for x in top_names_mse]

    # Sort by RMSE (real) and get top N
    data_sorted_rmse = sorted(data, key=lambda x: x[2])[:top_n]
    top_names_rmse = [x[0] for x in data_sorted_rmse]
    top_rmse_real = [x[2] for x in data_sorted_rmse]


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Top 10 by Test MSE (scaled)
    y_pos_1 = np.arange(len(top_names_mse))
    bars1 = ax1.barh(y_pos_1, top_mse_scaled, color='steelblue', alpha=0.8)
    ax1.set_yticks(y_pos_1)
    ax1.set_yticklabels(top_names_mse, fontsize=9)
    ax1.invert_yaxis()  # Best model at top
    ax1.set_xlabel('Test MSE (scaled)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} Models by Test MSE (Scaled)', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, top_mse_scaled)):
        ax1.text(val, i, f' {val:.4f}', va='center', fontsize=8)

    # Right plot: Top 10 by Test RMSE (real)
    y_pos_2 = np.arange(len(top_names_rmse))
    bars2 = ax2.barh(y_pos_2, top_rmse_real, color='coral', alpha=0.8)
    ax2.set_yticks(y_pos_2)
    ax2.set_yticklabels(top_names_rmse, fontsize=9)
    ax2.invert_yaxis()  # Best model at top
    ax2.set_xlabel('Test RMSE (real)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top {top_n} Models by Test RMSE (Real Scale)', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, top_rmse_real)):
        ax2.text(val, i, f' {val:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"TOP {top_n} MODELS SUMMARY")
    print(f"{'='*70}")

    # Best model by scaled MSE
    best_mse_name = top_names_mse[0]
    best_mse_metrics = results_dict[best_mse_name]
    print(f"\nBest Model (by scaled MSE): {best_mse_name}")
    print(f"  All Metrics:")
    for metric_name, metric_value in best_mse_metrics.items():
        if isinstance(metric_value, (np.floating, np.integer)):
            metric_value = float(metric_value)
        if 'scaled' in metric_name.lower():
            print(f"    - {metric_name}: {metric_value:.6f}")
        else:
            print(f"    - {metric_name}: {metric_value:,.2f}")

    # Best model by real RMSE
    best_rmse_name = top_names_rmse[0]
    best_rmse_metrics = results_dict[best_rmse_name]
    print(f"\nBest Model (by real RMSE): {best_rmse_name}")
    print(f"  All Metrics:")
    for metric_name, metric_value in best_rmse_metrics.items():
        if isinstance(metric_value, (np.floating, np.integer)):
            metric_value = float(metric_value)
        if 'scaled' in metric_name.lower():
            print(f"    - {metric_name}: {metric_value:.6f}")
        else:
            print(f"    - {metric_name}: {metric_value:,.2f}")

    print(f"{'='*70}\n")


# In[225]:


plot_top_models_comparison()


# ## Conclusions

# - Best score achieved by fine-tuned tf-idf single input model with stop words removed, learning rate scheduler and filling missing values with most frequent value
# - On 2 and 3  place are models with pretrained embeddings - if we use better embeddings and experiment we will probably be able to improve results and beat tf-idf approach, however time of generating embeddings is much longer than tf-idf vectorization

# ## New approach - trying to predict log of salary instead of standardized salary

# I updated functions to include option to log-transform the target variable and perform propere transformations when evaluating the model.

# In[28]:


LOG = True
log_models_dict = {}


# In[29]:


X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data(log=LOG)


# In[35]:


print(min(y_train), max(y_train))
print(min(y_valid), max(y_valid))
print(min(y_test), max(y_test))

print(y_train.shape, y_valid.shape, y_test.shape)


# In[30]:


preprocessing_results = preprocess_salary_data_with_tfidf(
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    categorical_columns=['Category', 'ContractType', 'ContractTime'],
    high_cardinality_columns=['Company', 'LocationNormalized', 'SourceName'],
    title_max_features=50,
    desc_max_features=800,
    batch_size=64,
    num_workers=0,
    desc_stop_words='english', # using stop words
    title_stop_words='english', # using stop words
    categorical_na_strategy='most_frequent',  
    high_card_na_strategy='most_frequent',    
    log=LOG,
)
train_loader = preprocessing_results['train_loader']
valid_loader = preprocessing_results['valid_loader']
test_loader = preprocessing_results['test_loader']

model = IntegratedNN(input_dim=preprocessing_results['feature_shape'][1]).to(device)
loss_fn = 'mse'
n_epochs = 25
optimizer_fn = 'adam'
patience = 3
lr = 0.001

trained_model, history, elapsed_time = train_single_input_model(
    model,
    train_loader,
    valid_loader,
    device=device,
    n_epochs=n_epochs,
    lr=lr,
    loss_fn=loss_fn,
    optimizer_fn=optimizer_fn,
    patience=patience,
    delta=0.001,
    early_stopping=None,
    use_lr_scheduler=True,
    scheduler_patience=0,
    scheduler_factor=0.5,
    seed=RANDOM_SEED,
    log=LOG
)

plot_losses_curves(history, loss_fn='mse')


# In[36]:


evaluate_single_input_model(
    model,
    test_loader,
    target_scaler=None,
    device=device,
    loss_fn_str='mse',
    model_name='tfidf_800_50_int_mf_bs64_adam_lrs_no_hid256_dr30_log',
    results_dict=log_models_dict,
    log=LOG
)


# 
