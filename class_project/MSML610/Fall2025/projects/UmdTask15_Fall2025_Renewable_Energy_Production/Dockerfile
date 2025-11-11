FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps (keep minimal)
RUN pip install --no-cache-dir \
    jupyter \
    pandas numpy \
    scikit-learn \
    matplotlib \
    requests

WORKDIR /work
EXPOSE 8888

# Default command: Jupyter (tokenless for grading convenience)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=", "--allow-root"]
