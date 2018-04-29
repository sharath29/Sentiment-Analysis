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