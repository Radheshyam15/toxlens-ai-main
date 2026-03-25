<<<<<<< HEAD
# ToxLens AI - Drug Toxicity Prediction System

A comprehensive AI-powered system for predicting drug toxicity and drug-drug interactions using Graph Neural Networks (GNNs).

## 🏗️ System Architecture

```
Data Sources → Dataset Preparation → GNN Training → Model Validation → Backend API → Frontend UI
     ↓              ↓                      ↓              ↓              ↓            ↓
  Tox21,         combined.csv         toxicity_gnn_model.pth     Flask API     Web Interface
 ClinTox,
  SIDER
```

## 📁 Project Structure

```
toxlens-ai-main/
├── data/                          # Raw and processed datasets
│   ├── tox21.csv                 # Tox21 toxicity dataset
│   ├── clintox.csv               # ClinTox dataset
│   ├── sider.csv                 # SIDER side effects dataset
│   └── combined.csv              # Combined balanced dataset
├── backend/                       # Flask API server
│   ├── app.py                    # Main Flask application
│   ├── requirements.txt          # Python dependencies
│   ├── model/                    # ML model components
│   │   ├── model.py              # GNN model architecture
│   │   ├── predict.py            # Prediction logic
│   │   └── toxicity_gnn_model.pth # Trained model weights
│   └── utils/                    # Utilities
│       └── reasoning.py          # LLM-powered reasoning
├── TOXLENS FRONTEND/             # Web interface
│   ├── index.html                # Main HTML page
│   ├── script.js                 # Frontend JavaScript
│   └── style.css                 # Styling
├── dataset.py                     # SMILES to graph conversion
├── prepare_dataset.py            # Dataset preparation script
├── train_gnn.py                  # Model training script
├── validate_model.py             # Model validation script
└── run_pipeline.py               # Pipeline runner
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd toxlens-ai-main
```

### 2. Install Dependencies
```bash
# Install Python packages
pip install -r backend/requirements.txt
```

### 3. Run the Complete Pipeline
```bash
# Basic pipeline (uses default hyperparameters)
python run_pipeline.py

# With hyperparameter tuning (recommended for best performance)
python run_pipeline.py --tune
```

This will:
- Prepare the combined dataset
- Tune hyperparameters (if --tune flag used or no saved parameters exist)
- Train the GNN model (if not already trained)
- Validate the model
- Start the backend server
- Open the frontend in your browser

### 4. Manual Execution

#### Prepare Dataset
```bash
python prepare_dataset.py
```

#### Tune Hyperparameters *[Optional]*
```bash
python tune_gnn.py
```

#### Train Model
```bash
python train_gnn.py
```

#### Validate Model
```bash
python validate_model.py
```

#### Start Backend
```bash
cd backend
python app.py
```

#### Open Frontend
Open `TOXLENS FRONTEND/index.html` in your web browser.

## 🔧 API Usage

### Prediction Endpoint
```bash
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "smiles1": "CCO",
  "smiles2": "C1=CC=C(C=C1)[N+](=O)[O-]"
}
```

Response:
```json
{
  "drugA": {
    "prediction": "NON-TOXIC",
    "confidence": 0.87
  },
  "drugB": {
    "prediction": "TOXIC",
    "confidence": 0.92
  },
  "interaction": "HIGH RISK ⚠️",
  "overall_confidence": 0.895,
  "structured_reasoning": {...},
  "reason": "Explanation text..."
}
```

## 🧠 Model Details

- **Architecture**: Graph Attention Network (GAT) with fingerprint integration
- **Input**: SMILES strings converted to molecular graphs
- **Features**: Atomic properties + Morgan fingerprints
- **Output**: Toxicity probability (0-1)
- **Training Data**: Balanced combination of Tox21, ClinTox, and SIDER datasets
## ⚙️ Hyperparameter Tuning

### Definition
Hyperparameter tuning is the process of choosing the best configuration settings for a machine learning model to improve its performance.

### Parameters vs Hyperparameters
| Type          | Meaning                              |
|---------------|--------------------------------------|
| Parameter     | Learned automatically (weights)      |
| Hyperparameter| Set manually before training         |

### Tuning Process
The project includes automated hyperparameter optimization using Optuna:

```bash
python tune_gnn.py
```

This optimizes:
- Learning rate
- Batch size
- Hidden dimensions
- Attention heads
- Dropout rate

The best hyperparameters are saved to `best_hyperparameters.json` and automatically loaded during training.
## 🎯 Key Features

- **Molecular Graph Processing**: Converts SMILES to graph representations
- **GNN-based Prediction**: Uses PyTorch Geometric for graph neural networks
- **Drug Interaction Analysis**: Assesses combined toxicity risk
- **Web Interface**: Modern, responsive UI for easy interaction
- **LLM Reasoning**: Optional AI-powered explanations (requires Ollama)

### 🧠 What is Hyperparameter Tuning?

**👉 Definition:**
Hyperparameter tuning is the process of choosing the best configuration settings for a machine learning model to improve its performance.

**⚙️ Difference**
| Type | Meaning |
|---|---|
| Parameter | Learned automatically (weights) |
| Hyperparameter | Set manually before training |

### 🕵️‍♂️ Model Interpretability

**👉 Definition:**
Model Interpretability (or Explainable AI) refers to the ability to understand and explain how a machine learning model makes its predictions, ensuring transparency and trust.

**⚙️ Difference**
| Concept | Meaning |
|---|---|
| Black Box Model | Predictions are made without clear reasoning (hard to trust). |
| Interpretable Model | The model provides transparent reasoning for *why* it made a prediction. |

### ⚡ High-Performance Computing Setup (Colab)

**👉 Definition:**
High-Performance Computing (HPC) involves using powerful cloud GPUs (like NVIDIA T4s) to process massive datasets and train complex deep learning models much faster than a standard laptop.

**⚙️ How to run ToxLens AI on Google Colab (Free GPU):**
Since tuning Graph Neural Networks is computationally heavy, you can easily deploy this pipeline on the cloud:

1. Open [Google Colab](https://colab.research.google.com/) and click **New Notebook**.
2. Go to the top menu: **Runtime** ➔ **Change runtime type** ➔ Select **T4 GPU**.
3. Paste and run this code block in the first cell:
   ```bash
   !git clone https://github.com/YOUR_GITHUB_USERNAME/toxlens-ai-main.git
   %cd toxlens-ai-main
   !pip install -r backend/requirements.txt
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install torch_geometric
   ```
4. Train your high-performance model:
   ```bash
   !python prepare_dataset.py
   !python tune_gnn.py
   !python train_gnn.py
   ```

## 📊 Datasets

- **Tox21**: 8,014 compounds with 12 toxicity assays
- **ClinTox**: 1,484 compounds with clinical toxicity data
- **SIDER**: 1,430 drugs with side effect information

## 🔄 Pipeline Flow

1. **Data Preparation** (`prepare_dataset.py`)
   - Load raw datasets
   - Convert to binary classification
   - Balance classes
   - Validate SMILES
   - Save combined dataset

2. **Graph Conversion** (`dataset.py`)
   - Parse SMILES with RDKit
   - Extract atom features
   - Build molecular graphs
   - Generate Morgan fingerprints

3. **Hyperparameter Tuning** (`tune_gnn.py`) *[Optional]*
   - Automated optimization using Optuna
   - Tune learning rate, batch size, hidden dims, etc.
   - Save best parameters to JSON

4. **Model Training** (`train_gnn.py`)
   - Load processed graphs
   - Train GAT-based GNN with optimized hyperparameters
   - Early stopping with validation
   - Save best model

5. **Prediction** (`backend/model/predict.py`)
   - Load trained model
   - Process input SMILES
   - Generate predictions
   - Assess interaction risk

6. **API Service** (`backend/app.py`)
   - Flask REST API
   - CORS enabled
   - Error handling

6. **Web Interface** (`TOXLENS FRONTEND/`)
   - HTML/CSS/JavaScript
   - Real-time predictions
   - Interactive visualizations

## ⚠️ Important Notes

- For research and academic use only
- Not for clinical diagnosis
- Model performance depends on training data quality
- LLM reasoning requires Ollama (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

=======
# ToxLens AI - Drug Toxicity Prediction System

A comprehensive AI-powered system for predicting drug toxicity and drug-drug interactions using Graph Neural Networks (GNNs).

## 🏗️ System Architecture

```
Data Sources → Dataset Preparation → GNN Training → Model Validation → Backend API → Frontend UI
     ↓              ↓                      ↓              ↓              ↓            ↓
  Tox21,         combined.csv         toxicity_gnn_model.pth     Flask API     Web Interface
 ClinTox,
  SIDER
```

## 📁 Project Structure

```
toxlens-ai-main/
├── data/                          # Raw and processed datasets
│   ├── tox21.csv                 # Tox21 toxicity dataset
│   ├── clintox.csv               # ClinTox dataset
│   ├── sider.csv                 # SIDER side effects dataset
│   └── combined.csv              # Combined balanced dataset
├── backend/                       # Flask API server
│   ├── app.py                    # Main Flask application
│   ├── requirements.txt          # Python dependencies
│   ├── model/                    # ML model components
│   │   ├── model.py              # GNN model architecture
│   │   ├── predict.py            # Prediction logic
│   │   └── toxicity_gnn_model.pth # Trained model weights
│   └── utils/                    # Utilities
│       └── reasoning.py          # LLM-powered reasoning
├── TOXLENS FRONTEND/             # Web interface
│   ├── index.html                # Main HTML page
│   ├── script.js                 # Frontend JavaScript
│   └── style.css                 # Styling
├── dataset.py                     # SMILES to graph conversion
├── prepare_dataset.py            # Dataset preparation script
├── train_gnn.py                  # Model training script
├── validate_model.py             # Model validation script
└── run_pipeline.py               # Pipeline runner
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd toxlens-ai-main
```

### 2. Install Dependencies
```bash
# Install Python packages
pip install -r backend/requirements.txt
```

### 3. Run the Complete Pipeline
```bash
# Basic pipeline (uses default hyperparameters)
python run_pipeline.py

# With hyperparameter tuning (recommended for best performance)
python run_pipeline.py --tune
```

This will:
- Prepare the combined dataset
- Tune hyperparameters (if --tune flag used or no saved parameters exist)
- Train the GNN model (if not already trained)
- Validate the model
- Start the backend server
- Open the frontend in your browser

### 4. Manual Execution

#### Prepare Dataset
```bash
python prepare_dataset.py
```

#### Tune Hyperparameters *[Optional]*
```bash
python tune_gnn.py
```

#### Train Model
```bash
python train_gnn.py
```

#### Validate Model
```bash
python validate_model.py
```

#### Start Backend
```bash
cd backend
python app.py
```

#### Open Frontend
Open `TOXLENS FRONTEND/index.html` in your web browser.

## 🔧 API Usage

### Prediction Endpoint
```bash
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "smiles1": "CCO",
  "smiles2": "C1=CC=C(C=C1)[N+](=O)[O-]"
}
```

Response:
```json
{
  "drugA": {
    "prediction": "NON-TOXIC",
    "confidence": 0.87
  },
  "drugB": {
    "prediction": "TOXIC",
    "confidence": 0.92
  },
  "interaction": "HIGH RISK ⚠️",
  "overall_confidence": 0.895,
  "structured_reasoning": {...},
  "reason": "Explanation text..."
}
```

## 🧠 Model Details

- **Architecture**: Graph Attention Network (GAT) with fingerprint integration
- **Input**: SMILES strings converted to molecular graphs
- **Features**: Atomic properties + Morgan fingerprints
- **Output**: Toxicity probability (0-1)
- **Training Data**: Balanced combination of Tox21, ClinTox, and SIDER datasets
## ⚙️ Hyperparameter Tuning

### Definition
Hyperparameter tuning is the process of choosing the best configuration settings for a machine learning model to improve its performance.

### Parameters vs Hyperparameters
| Type          | Meaning                              |
|---------------|--------------------------------------|
| Parameter     | Learned automatically (weights)      |
| Hyperparameter| Set manually before training         |

### Tuning Process
The project includes automated hyperparameter optimization using Optuna:

```bash
python tune_gnn.py
```

This optimizes:
- Learning rate
- Batch size
- Hidden dimensions
- Attention heads
- Dropout rate

The best hyperparameters are saved to `best_hyperparameters.json` and automatically loaded during training.
## 🎯 Key Features

- **Molecular Graph Processing**: Converts SMILES to graph representations
- **GNN-based Prediction**: Uses PyTorch Geometric for graph neural networks
- **Drug Interaction Analysis**: Assesses combined toxicity risk
- **Web Interface**: Modern, responsive UI for easy interaction
- **LLM Reasoning**: Optional AI-powered explanations (requires Ollama)

### 🧠 What is Hyperparameter Tuning?

**👉 Definition:**
Hyperparameter tuning is the process of choosing the best configuration settings for a machine learning model to improve its performance.

**⚙️ Difference**
| Type | Meaning |
|---|---|
| Parameter | Learned automatically (weights) |
| Hyperparameter | Set manually before training |

### 🕵️‍♂️ Model Interpretability

**👉 Definition:**
Model Interpretability (or Explainable AI) refers to the ability to understand and explain how a machine learning model makes its predictions, ensuring transparency and trust.

**⚙️ Difference**
| Concept | Meaning |
|---|---|
| Black Box Model | Predictions are made without clear reasoning (hard to trust). |
| Interpretable Model | The model provides transparent reasoning for *why* it made a prediction. |

### ⚡ High-Performance Computing Setup (Colab)

**👉 Definition:**
High-Performance Computing (HPC) involves using powerful cloud GPUs (like NVIDIA T4s) to process massive datasets and train complex deep learning models much faster than a standard laptop.

**⚙️ How to run ToxLens AI on Google Colab (Free GPU):**
Since tuning Graph Neural Networks is computationally heavy, you can easily deploy this pipeline on the cloud:

1. Open [Google Colab](https://colab.research.google.com/) and click **New Notebook**.
2. Go to the top menu: **Runtime** ➔ **Change runtime type** ➔ Select **T4 GPU**.
3. Paste and run this code block in the first cell:
   ```bash
   !git clone https://github.com/YOUR_GITHUB_USERNAME/toxlens-ai-main.git
   %cd toxlens-ai-main
   !pip install -r backend/requirements.txt
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install torch_geometric
   ```
4. Train your high-performance model:
   ```bash
   !python prepare_dataset.py
   !python tune_gnn.py
   !python train_gnn.py
   ```

## 📊 Datasets

- **Tox21**: 8,014 compounds with 12 toxicity assays
- **ClinTox**: 1,484 compounds with clinical toxicity data
- **SIDER**: 1,430 drugs with side effect information

## 🔄 Pipeline Flow

1. **Data Preparation** (`prepare_dataset.py`)
   - Load raw datasets
   - Convert to binary classification
   - Balance classes
   - Validate SMILES
   - Save combined dataset

2. **Graph Conversion** (`dataset.py`)
   - Parse SMILES with RDKit
   - Extract atom features
   - Build molecular graphs
   - Generate Morgan fingerprints

3. **Hyperparameter Tuning** (`tune_gnn.py`) *[Optional]*
   - Automated optimization using Optuna
   - Tune learning rate, batch size, hidden dims, etc.
   - Save best parameters to JSON

4. **Model Training** (`train_gnn.py`)
   - Load processed graphs
   - Train GAT-based GNN with optimized hyperparameters
   - Early stopping with validation
   - Save best model

5. **Prediction** (`backend/model/predict.py`)
   - Load trained model
   - Process input SMILES
   - Generate predictions
   - Assess interaction risk

6. **API Service** (`backend/app.py`)
   - Flask REST API
   - CORS enabled
   - Error handling

6. **Web Interface** (`TOXLENS FRONTEND/`)
   - HTML/CSS/JavaScript
   - Real-time predictions
   - Interactive visualizations

## ⚠️ Important Notes

- For research and academic use only
- Not for clinical diagnosis
- Model performance depends on training data quality
- LLM reasoning requires Ollama (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

>>>>>>> 925dc03d (Add missing files)
This project is licensed under the MIT License - see the LICENSE file for details.