# Parallel Unlearning in Inherited Model Networks

This repository contains the official implementation of the **Parallel Unlearning in Inherited Model Networks** framework described in the paper *“Fishers Harvest Parallel Unlearning in Inherited Model Networks”* (arXiv:2408.08493). :contentReference[oaicite:0]{index=0}

Machine unlearning aims to remove the influence of specific training data from trained models **as if the data were never seen**, while preserving performance on the remaining data. In inheritance-oriented model networks (e.g., federated learning, incremental learning, transfer learning), unlearning is especially challenging due to complex model dependencies. :contentReference[oaicite:1]{index=1}

---

## 📌 Key Contributions

This project implements the core components of the proposed framework:

- **Unified Model Inheritance Graph (UMIG)**:  
  A Directed Acyclic Graph (DAG) abstraction to represent inheritance relationships among models. :contentReference[oaicite:2]{index=2}

- **Fisher Inheritance Unlearning (FIUn)**:  
  An unlearning method leveraging the Fisher Information Matrix (FIM) to identify and adjust parameters associated with unlearned knowledge across inherited models. :contentReference[oaicite:3]{index=3}

- **Merging FIM (MFIM) Module**:  
  Efficiently merges Fisher information from multiple upstream models, enabling **fully parallel unlearning** and reducing computational overhead. :contentReference[oaicite:4]{index=4}

---

## 🚀 Features

- 📈 **Parallel unlearning** across large, inheritance-linked model networks  
- 🧠 Efficient unlearning via Fisher Information Matrix  
- 🔗 DAG-based representation of inheritance chains  
- 📊 Scripted evaluation on both single-class and multi-class unlearning tasks

---

## 🔧 Installation

> ⚠️ *Fill in with your project’s actual dependencies.*

```bash
git clone https://github.com/MJLee00/Parallel-Unlearning-in-Inherited-Model-Networks.git
cd Parallel-Unlearning-in-Inherited-Model-Networks

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Linux/macOS
venv\Scripts\activate     # on Windows

# Install requirements
pip install -r requirements.txt
