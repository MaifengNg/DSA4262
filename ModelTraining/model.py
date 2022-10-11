"""
model.py serves as a utility module for DSA4262
"""
import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample
from utils import load_data_json_file
from utils import load_info_json_file


def concat_two_dataframe(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas dataframe loaded by concating two dataframes
    and removing the first two dataframe.

    :param df_1 pd.DataFrame
    :param df_2 pd.DataFrame
    :return pandas dataframe
    """
    processed_dataframe = pd.concat([
        df_1,
        df_2],
        axis=1,
        join="inner")
    processed_dataframe = processed_dataframe.iloc[
        :,
        2:]
    return processed_dataframe


def upsample_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas dataframe where the labels have equal sizes

    :param pd.DataFrame data_frame: data_frame to upsample
    :return pandas dataframe
    """
    df_label_counts = data_frame['label'].value_counts()
    min_label = 0
    max_label = 1
    max_sample_size = df_label_counts[1]
    if df_label_counts[0] > df_label_counts[1]:
        min_label = 1
        max_label = 0
        max_sample_size = df_label_counts[0]

    df_majority = data_frame[(data_frame['label'] == max_label)]
    df_minority = data_frame[(data_frame['label'] == min_label)]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=max_sample_size,
                                     random_state=42)

    df_equalized = pd.concat([df_minority_upsampled, df_majority])
    return df_equalized


def train_save_model(data_frame: pd.DataFrame, path_to_save: str) -> None:
    """
    :param pd.DataFrame data_frame: data_frame to upsample
    :param str path_to_save: path to logistic regression model
    :return None
    """
    X = data_frame.drop(
        ['label', 'transcript_id', 'transcript_position'], axis=1)
    y = data_frame['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)

    logreg_model = LogisticRegression(random_state=0)
    logreg_model.fit(X_train, y_train)
    y_pred = logreg_model.predict(X_test)
    print(
        f'Accuracy of logistic regression classifier on test set: {logreg_model.score(X_test, y_test)}')
    print(classification_report(y_test, y_pred))
    pickle.dump(logreg_model, open(path_to_save, 'wb'))

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    JSON_DATA_PATH = f'{dir_path}/Data/data.json'
    json_data_dataframe = load_data_json_file(JSON_DATA_PATH)

    JSON_INFO_PATH = f'{dir_path}/Data/data.info'
    data_info_dataframe = load_info_json_file(JSON_INFO_PATH)

    concat_dataframe = concat_two_dataframe(
        data_info_dataframe, json_data_dataframe)
    PATH_SAVE = f'{dir_path}/Data/data_processed.csv'
    concat_dataframe.to_csv(PATH_SAVE, index=False)

    upsampled_dataframe = upsample_dataframe(concat_dataframe)
    PATH_SAVE_MODEL = f'{dir_path}/Data/logreg_model.sav'
    train_save_model(upsampled_dataframe, PATH_SAVE_MODEL)
