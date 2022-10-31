# DSA4262

## Steps to train 

### Step 1: Clone repository to local machine
```
Git clone repository to local machine
```

### Step 2: Turn on python3 virtual environment
```
Turn on python3 virtual environment

source venv/bin/activate
```

### Step 3: Install requirements
```
Install requirements

pip3 install -r requirements.txt
```

### Step 4: Training a classifier 
```
The train.py script in ModelTraining is written to train a classifier to perform classification on the RNA-Seq dataset.

There are 3 required arguments and 1 optional arguments to train a classifier.

1. data: The directory containing the training dataset
2. label: The directory containing the label dataset
3. model_dir: The directory to save the model after training.
4. standardize: To standardize the training dataset to have mean 0 and variance 1.

Below is an example of how to run the train.py script.
python3 ModelTraining/train.py --data /Data/data.json --label /Data/data.info --model_dir /Model --standardize TRUE
```

### Step 5: Testing model trained
```
The test.py script in ModelTraining is written to test a classifier to perform classification on the RNA-Seq dataset.

There are 3 required arguments to test a trained classifier.

1. data: The directory containing the test dataset
2. model_dir: The directory containing the model saved from train.py
3. save: The directory to save the results 

Below is an example of how to run the test.py script.
python3 ModelTraining/test.py --data /SampleData/sample.json --model_dir /Model/model.sav --save /Results
```