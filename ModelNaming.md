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
| **5.** | **(LRS)** | **Learning Rate Scheduler:** Specifies if a scheduler is used. | `lrs_no` (No scheduler), or a specific scheduler code (e.g., `lrs_cos`) |
| **6.** | **hid(H)** | **First Hidden Layer Size:** The number of neurons in the first hidden layer. | `hid64`, `hid128`, `hid256` |
| **7.** | **dr(D)** | **Dropout Rate:** The dropout probability applied (as a percentage). **Recommended format: `dr(XX)` where XX is the percentage.** | `dr30` (for 0.3/30%), `dr50` (for 0.5/50%) |
| **8.** | **(Optional)** | **Run/Version Info:** An optional field for unique identification. | e.g., `v1`, `run2` |

**Example:** `cat_mf_bs64_adam_lrs_no_hid128_dr30_v1`

***

### 2. Naming Structure for Text Data

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
| **6.** | **(LRS)** | **Learning Rate Scheduler:** Specifies if a scheduler is used. | `lrs_no`, or a specific scheduler code |
| **7.** | **hid(H)** | **First Hidden Layer Size:** The number of neurons in the first hidden layer. | `hid64`, `hid128`, `hid256` |
| **8.** | **dr(D)** | **Dropout Rate:** The dropout probability applied (as a percentage). | `dr30` (for 0.3/30%), `dr50` (for 0.5/50%) |
| **9.** | **(Optional)** | **Run/Version Info:** An optional field for unique identification. | e.g., `v1`, `run2` |

***
