# Forecasting Renewable Energy Production (UmdTask15)

This project demonstrates a clean, reproducible scaffold for forecasting renewable energy production. It includes a minimal API, a worked example, and a small utilities module for loading data, feature engineering, modeling, and evaluation.

## How to run (locally with Docker)
```bash
# from repo root
docker build -t re-forecast:latest class_project/MSML610/Fall2025/projects/UmdTask15_Fall2025_Renewable_Energy_Production
docker run --rm -p 8888:8888 \
  -v "$PWD":/work -w /work \
  re-forecast:latest \
  jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token= --allow-root

