# Dockerfile

# We use a python 3.7 image
FROM python:3.7

# Allows docker to cache installed dependencies between builds
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Install vim
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

# create directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app

# run python detect_new.py
CMD ["python", "test.py", "--data", "/Sample/sample.json", "--model_dir", "/Model/model.sav",  "--save", "/Results",  "--save_file_name", "results_sample"]
# CMD ["python", "train.py", "--data", "/Data/data.json", "--model_dir", "/Model",  "--label", "/Data/data.info"]