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
python3 ModelTraining/model.py --data /Data/data.json --label /Data/data.info --model_dir /Model --standardize TRUE  
```

### Step 5: Testing model trained
```
python3 ModelTraining/test.py --data /SampleData/sample.json --model_dir /Model/model.sav --save /Results
```