## Model Naming Convention

This convention is designed to systematically encode all critical hyperparameters and preprocessing choices directly into the model file name, ensuring experiments are fully reproducible and easy to track.

### 1. Naming Structure for Categorical Data (No Text)

This structure is used when the model is trained *without* any text features.

**Format:**
$$\text{cat}\_\text{(MVI)}\_\text{bs}\text{(B)}\_\text{(Opt)}\_\text{(LRS)}\_\text{hid}\text{(H)}\_\text{dr}\text{(D)}$$

| Position | Code | Description | Options |
| :--- | :--- | :--- | :--- |
| **1.** | **cat** | **Data Type:** Indicates the model uses only categorical/numerical features. | `cat` |
| **2.** | **(MVI)** | **Missing Values Imputation:** Strategy for handling missing data. | `unk` (Unknown/Placeholder), `mf` (Mean/Median/Mode Fill) |
| **3.** | **bs(B)** | **Batch Size:** The training batch size. | `bs32`, `bs64` |
| **4.** | **(Opt)** | **Optimizer:** The training optimization algorithm. | `adam`, `sgd` |
| **5.** | **(LRS)** | **Learning Rate Scheduler:** Specifies if a scheduler is used. | `lrs_no` (No scheduler),`lrs` - reduce LR (by default 0.5 every 2 epochs without improvement) |
| **6.** | **hid(H)** | **First Hidden Layer Size:** The number of neurons in the first hidden layer. | `hid64`, `hid128`, `hid256` |
| **7.** | **dr(D)** | **Dropout Rate:** The dropout probability applied (as a percentage). **Recommended format: `dr(XX)` where XX is the percentage.** | `dr30` (for 0.3/30%), `dr50` (for 0.5/50%) |
| **8.** | **(Optional)** | **Run/Version Info:** An optional field for unique identification. | e.g., `v1`, `run2` |

**Example:** `cat_mf_bs64_adam_lrs_no_hid128_dr30_v1`

***

### 2. Naming Structure for Text Data - Embeddings from Sentence Transformers or TF-IDF (First Part of Project)

This structure is used when the model includes text features, which are processed using either pretrained embeddings or TF-IDF.



**Format:**
$$\text{(TextPrep)}\_\text{(ModelType)}\_\text{(MVI)}\_\text{bs}\text{(B)}\_\text{(Opt)}\_\text{(LRS)}\_\text{hid}\text{(H)}\_\text{dr}\text{(D)}\_\text{(Optional)}$$

| Position | Code | Description | Options |
| :--- | :--- | :--- | :--- |
| **1.** | **(TextPrep)** | **Text Preprocessing:** Method used to vectorize text data. | `emb` (Pretrained Embeddings), `tfidf` (TF-IDF vectorizer) |
| **1a.** | **tfidf\_(T)** | **TF-IDF Details:** Specific parameters when using TF-IDF. | `tfidf_800_50`, `tfidf_svd_200`, `tfidf_stop_words` |
| **2.** | **(ModelType)** | **Model Architecture Type:** The high-level neural network structure. | **`sr`**, **`srbn`**, **`int`**, **`multi`** |
| **3.** | **(MVI)** | **Missing Values Imputation:** Strategy for handling missing data. | `unk` (Unknown/Placeholder), `mf` (Mean/Median/Mode Fill) |
| **4.** | **bs(B)** | **Batch Size:** The training batch size. | `bs32`, `bs64` |
| **5.** | **(Opt)** | **Optimizer:** The training optimization algorithm. | `adam`, `sgd` |
| **6.** | **(LRS)** | **Learning Rate Scheduler:** Specifies if a scheduler is used. | `lrs_no` (No scheduler),`lrs` - reduce LR (by default 0.5 every 2 epochs without improvement) |
| **7.** | **hid(H)** | **First Hidden Layer Size:** The number of neurons in the first hidden layer. | `hid64`, `hid128`, `hid256` |
| **8.** | **dr(D)** | **Dropout Rate:** The dropout probability applied (as a percentage). | `dr30` (for 0.3/30%), `dr50` (for 0.5/50%) |
| **9.** | **(Optional)** | **Run/Version Info:** An optional field for unique identification. | e.g., `v1`, `run2` |

***

### 3. Naming Structure for Text Data - Self-Taught Embeddings, Word2Vec, FastText (Second Part of Project)

**Assumptions (Fixed Parameters - Not Included in Names)**
- **Optimizer:** Adam
- **Batch Size:** 64
- **Model Type:** Multi (multimodal architecture)
- **Missing Values Imputation:** Unknown/Placeholder strategy
- **Learning rate:** By default 0.001, if other then information at the end of name of the model is provided

**Format:**
$$\text{emb}\_\text{(EmbSize)}\_\text{mlen}\text{(MaxLen)}\_\text{mf}\text{(MinFreq)}\_\text{(Arch)}\_\text{(EmbMode)}\_\text{chid}\text{(CH)}\_\text{rhid}\text{(RH)}\_\text{dr}\text{(D)}$$

| Position | Code | Description | Options |
|:---------|:-----|:------------|:--------|
| **1.** | **emb** | **Embedding Type:** Indicates self-trained embeddings are used |  `word2vec`, `fasttext`, `selftaught` |
| **2.** | **(EmbSize)** | **Embedding Dimension:** Size of the word embedding vectors | `e50`, `e100`, `e128`, `e256`, `e300` |
| **3.** | **mlen(MaxLen)** | **Max Text Length:** Maximum number of tokens per text sequence | `mlen50`, `mlen100`, `mlen150`, `mlen200` |
| **4.** | **mf(MinFreq)** | **Min Frequency in Vocab:** Minimum word frequency to be included in vocabulary | `mf2`, `mf3`, `mf5`, `mf10` |
| **5.** | **(Arch)** | **Model Architecture:** Base network structure | `base` (Basic NN), `cnn` (CNN layers), `base_res` (Base with residual), `cnn_res` (CNN with residual) |
| **6.** | **(EmbMode)** | **Embedding Training Mode:** How embeddings are handled during training | `learn` (Learn embeddings during training - for `selftaught`), `pass_emb` (Pass pre-computed embeddings), `froz_tok` (Frozen embedding matrix), `unfroz_tok` (Trainable embedding matrix) |
| **7.** | **chid(CH)** | **Categorical Hidden Size:** Hidden layer size for categorical/numerical features | `chid64`, `chid128`, `chid256`, `chid512` |
| **8.** | **rhid(RH)** | **Regressor Hidden Size:** Hidden layer size for final regression head | `rhid32`, `rhid64`, `rhid128`, `rhid256` |
| **9.** | **dr(D)** | **Dropout Rate:** Dropout probability (as percentage) | `dr20`, `dr30`, `dr40`, `dr50` |
| **10.** | **(LRS)** | **Learning Rate Scheduler:** Specifies if a scheduler is used. | `lrs_no` (No scheduler),`lrs` - reduce LR (by default 0.5 every 2 epochs without improvement) |
| **11.** | **(Optional)** | **Run/Version Info:** Optional unique identifier | `v1`, `v2`, `run1`, `exp3` |

#### Architecture Types Explained

| Code | Description |
|:-----|:------------|
| `base` | Basic feedforward network with dense layers |
| `cnn` | Convolutional layers for text processing |
| `base_res` | Basic architecture with residual connections |
| `cnn_res` | CNN architecture with residual connections |

#### Embedding Mode Details

| Code | Description | Use Case |
|:-----|:------------|:---------|
| `pass_emb` | Pre-computed embeddings passed as input | No embedding layer in model; uses external embeddings |
| `froz_tok` | Token indices with frozen embedding matrix | Embedding layer with requires_grad=False |
| `train_tok` | Token indices with trainable embedding matrix | Embedding layer with requires_grad=True; fine-tuning |

