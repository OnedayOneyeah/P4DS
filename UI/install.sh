#!/bin/bash
conda create -y -n p4ds python=3.11
source ~/anaconda3/etc/profile.d/conda.sh
conda activate p4ds

conda install -y -c conda-forge faiss-cpu

pip install "codeinterpreterapi[all]"
pip install chromadb
conda install -c conda-forge faiss-cpu
pip install sentence-transformers PyPDF2
pip install -U langchain-community
pip install langchain faiss-cpu
pip install pypdf
pip install tavily-python
pip install PyMuPDF

# 충돌 패키지 제거
pip uninstall -y pydantic pydantic-settings pydantic-core langchain langchain-core langchain-community

# pydantic v1 및 langchain 안정 버전 재설치
pip install "pydantic<2.0"
pip install "pydantic-settings<2.0"
pip install "langchain==0.0.350"
pip install codeinterpreterapi==0.1.20

# 추가 (정문)
pip install pandas sentence_transformers openpyxl streamlit python-dotenv