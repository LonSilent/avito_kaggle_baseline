# avito_kaggle_baseline

## Prepare your data

Download dataset from [kaggle](https://www.kaggle.com/c/avito-demand-prediction/data)

## Dependencies
`numpy`, `scipy`, `scikit-learn`, `xgboost`, `nltk`

## Download image prediction

We apply image label prediction task with [Resnet.](https://keras.io/applications/#resnet50) You can download results from [here](https://drive.google.com/file/d/1zi9vXs9Z8-Yaeq4Ca3mA1ao5t-6-xRaH/view?usp=sharing) and unzip it.

## Run

```
python3 baseline.py train.csv test.csv
```

## Add image prediciton feature

```
python3 add_image_pred.py train.csv test.csv image_pred/
```