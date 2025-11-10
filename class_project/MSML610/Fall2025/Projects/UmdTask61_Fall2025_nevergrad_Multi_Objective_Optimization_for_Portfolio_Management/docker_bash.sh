#!/usr/bin/env bash
docker run --rm -it -v "$(pwd)":/workspace -w /workspace portfolio-project:latest bash
