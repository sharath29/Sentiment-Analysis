# SC-Sentiment-Analysis
Using Concolutoinal neural network to identify emotions from facial images.
Fer 2013 data set used for training.


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://github.com/ignaciorlando/skinner/wiki/Keras-and-TensorFlow-installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Accuracy

![Accuracy Plot](https://raw.githubusercontent.com/sharath29/SC-Sentiment-Analysis/master/results.png)
![Accuracy values](https://raw.githubusercontent.com/sharath29/SC-Sentiment-Analysis/master/results_values.png)

Accuracy values represented above:
Accuracy: ratio of total correct prediction by total data used for prediction. i.e. correct predictions / total input data/
Precision: ratio of true positive to sum of true positive and false positive. True Positives / (True Positives + False Positives).
Recall: Recall is calculated as True Positives / (True Positives + False Negatives).
## Dataset

- [Kraggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

## Usage

1.Download the data set and put it into data folder.

2.Run
```
python emotion_recognition.py run
```

3.Train
```
python emotion_recognition.py train
```

4.Plots are stored in ipython notebook. Open in jupyter notebook.