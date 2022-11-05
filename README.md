# DSA4262

## Steps to train 

### Step 1: Clone repository to local machine
```
Git clone repository to local machine
```

### Step 2(optional): Install pip3
```
Install pip3 in order to install requirements to run the test script.

sudo apt install python3-pip
```

### Step 3: Install requirements
```
Install requirements

pip3 install -r requirements.txt
```

### Step 4: Training a classifier 
```
The train.py script in ModelTraining is written to train a classifier to perform classification on the RNA-Seq dataset.

There are 3 required arguments to train a classifier.
1. data: The directory containing the training dataset
2. label: The directory containing the label dataset
3. model_dir: The directory to save the model after training.

There is a 4th optional argument to train a classifier.
4. standardize: To standardize the training dataset to have mean 0 and variance 1.

Below is an example of how to run the train.py script.
python3 train.py --data /Data/data.json --label /Data/data.info --model_dir /Model 
```

### Step 5: Testing model trained
```
The test.py script in ModelTraining is written to test a classifier to perform classification on the RNA-Seq dataset.

There are 3 required arguments to test a trained classifier.
1. data: The directory containing the test dataset
2. model_dir: The directory containing the model saved from train.py
3. save: The directory to save the results 

There is a 4th optional argument to train a classifier.
4. save_file_name: The file name to save the results of the classification. The file extension has been excluded. Do not include it.

Below is an example of how to run the test.py script.
python3 test.py --data /Sample/sample.json --model_dir /Model/model.sav --save /Results --save_file_name sample
```

### Testing the model on a sample data
```
The command 
python3 test.py --data /Sample/sample.json --model_dir /Model/model.sav --save /Results --save_file_name sample 
will run the model on the sample.json sample data.
```


## Docker solution
```
If you have docker installed, you can run the command `bash start_script.sh`.
This bash script helps you run the test.py script on the sample.json data.
```