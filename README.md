Heart Disease Prediction
-----------------------

Predict whether or not a person is suffering from heart disease. It's predicted value tells the  about angiographic disease status as value 0: < 50% diameter narrowing and value 1: > 50% diameter narrowing .  
*  A brief information about angiographic disease status [here].(https://www.ncbi.nlm.nih.gov/pubmed/22045968)
*  The dataset is taken from https://archive.ics.uci.edu/ml/datasets/heart+Disease  .

Installation
----------------------

### Download the data

* Download `predict.py` and `uci-heart-disease/data`.
* Open `predict.py` and go to `line-52`.
* Switch the file address to the file address of downloaded `uci-heart-disease/data` on your computer and save it.

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 3.
    * You may want to use a virtual environment for this.

Usage
-----------------------

* Run `predict.py`.
    * This will tell the accuracy of the model first and then asks about person.
        * Age (should be natural number)
        * Sex (1 for male ,and 0 for female)
        * fbs (fasting blood suger , 1 : fbs > 120 and 0 : fbs<120)
        * thalach (Maximum heart rate achieved in natural number)
        * cp (Chest pain in range of 1 to 4)
        * exang (Exercise induced angina 1 : yes and 0 : no)

Extending this
-------------------------

If you want to extend this work, here are a few places to start:

* Download `Heart Disease prediction without medical support (kaggle notebook).ipynb`.
* We can use another algorithms for another accuracies as shown in this file.
