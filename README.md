# Color season detector
This is a deep learning / machine learning project, intended to explore the possibilities of seasonal analysis. 
As a result, there is an application which provides the averaged probabilities of subseasons and apply a colorful virtual draping to your photos. 


# Presentation
Full description of work:
https://canva.link/nlbrbn3cqdbt72e


# How-to run
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
**access from web browser Local URL**

Enjoy!


# Training data and model comparison
`season_classification.ipynb` contains a full colab notebook which has been used for training and model verifications.
Please check all results here:

https://colab.research.google.com/drive/1EjoUpEYlEzEt_wKivrD8dvSYA5oh-XTW?usp=sharing


# Dataset
The existing (old) dataset is accessible here:

https://github.com/lorenzo-stacchio/Deep-Armocromia

`all.zip` contains my dataset (new): 

https://drive.google.com/drive/folders/1hrs-jSGaW92PYLcqQvVCd2CZSvxG4d3t?usp=sharing
This new dataset was gathered from publicly available images on the Internet.
600 images, 50 images per one of each season. 


# Structure
`models/` contains models for seasonal classification. See the presentation for a detailed explanation.

`season_classification.ipynb` contains main code for model training

`season_features.py` contains some common code for face segmentation and feature extraction

`app.py` is a streamlit single-page web application - the final product which uses the models

`preprocess_utils/` contains several scripts to detect and remove duplicates: `check_duplicate.py`,  `check_self_duplicates.py`
