#!/usr/bin/env bash
docker run --rm -it -p 8888:8888 -v "$(pwd)":/workspace -w /workspace portfolio-project:latest jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
