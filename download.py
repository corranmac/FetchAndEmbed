# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import pipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    !git clone https://github.com/corranmac/open_retrieval
    !git clone https://github.com/allenai/vila
    %cd /content/open_retrieval 
    !pip install -r /content/open_retrieval/requirements.txt
    %cd /content/vila 
    !pip install -e . -q # Install the `vila` library 
    !pip install -r requirements.txt -q# Only install the dependencies 
    !sudo apt-get install -y poppler-utils -q
    !pip install layoutparser -U -q
    !pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2' -q
    !pip install sentence-transformers sentence-splitter -q
    !pip install pymupdf

if __name__ == "__main__":
    download_model()
