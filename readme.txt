How to reproduce the experiments on NIH dataset.

1. please download the data from the official NIH dataset website.
https://nihcc.app.box.com/v/ChestXray-NIHCC

2. please extract and put the images into the `images/` directory.

3. install the required packages
```
pip install -r requirements.txt
```

3. For training, run
```
python train.py
```

For the demo of training an epoch, please refer to demo/train.mp4
The average time of training and evaluating an epoch is about 26 minutes.

4. For evaluation, run
```
python test.py
```

The test script should produce the AUC score of 0.78584
Please refer to the demo/test.mp4


Our specs
cpu: Intel(R) Core(TM) i5-10500 CPU @ 3.10GHz
gpu: NVIDIA GeForce GTX 1080 Ti
ram: 24GB