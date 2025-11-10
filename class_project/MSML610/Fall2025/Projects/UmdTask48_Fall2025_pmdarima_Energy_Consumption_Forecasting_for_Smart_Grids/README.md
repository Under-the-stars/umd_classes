# Energy Consumption Forecasting for Smart Grids (PMDARIMA)

## 📘 Objective
Develop a forecasting model to predict **hourly energy consumption** for a smart grid system using the **PMDARIMA** library.  
The goal is to optimize the model for accuracy while handling large-scale and noisy time-series data.

---

## 📊 Dataset
**UCI Machine Learning Repository:**  
[Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

- Time range: 2006–2010  
- Sampling rate: 1 minute (aggregated to hourly)  
- Target variable: `Global_active_power`

---

## ⚙️ Project Structure
class_project/MSML610/Fall2025/Projects/UmdTask48_Fall2025_pmdarima_Energy_Consumption_Forecasting_for_Smart_Grids/
│
├── pmdarima.API.ipynb # PMDARIMA API demonstration
├── pmdarima.API.md # API documentation and explanation
├── pmdarima.example.ipynb # End-to-end forecasting example
├── pmdarima.example.md # Example explanation and evaluation
├── pmdarima_utils.py # Utility functions for data loading and metrics
├── requirements.txt # Python dependencies
├── Dockerfile # Environment configuration
├── docker_build.sh # Script to build Docker image
├── docker_jupyter.sh # Script to launch Jupyter in container
├── docker_bash.sh # Optional: start an interactive container shell
└── README.md # Project overview

## 🧠 Run Instructions
1. **Build Docker Image**
   ```bash
   bash docker_build.sh
   
2. Run Jupyter Notebook
   ```bash
   bash docker_jupyter.sh

3. Open in Browser
   Navigate to http://localhost:8888

4. Run Notebooks
   Open:
   pmdarima.API.ipynb – learn and visualize PMDARIMA usage
   pmdarima.example.ipynb – run the complete forecasting pipeline

🧮 Current Progress
✅ Project initialized
✅ Data loading and helper functions implemented
✅ Model training and forecasting completed
📊 Evaluation and fine-tuning planned (next phase)

📦 Dependencies
numpy
pandas
matplotlib
scikit-learn
statsmodels
pmdarima

✍️ Author
Tanuka Majumder
University of Maryland – MSML610 (Fall 2025)