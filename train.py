"""
model.py serves as a scrip to train a logistic regression model
"""
import argparse
import os
import pickle
import pandas as pd
from utils import one_hot_encode_nucleotide_dataframe
from utils import load_data_json_file
from utils import load_info_json_file
from utils import concat_two_dataframe
from utils import drop_columns_from_dataframe
from utils import upsample_dataframe
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def train_save_model(data_frame: pd.DataFrame, path_to_save: str) -> None:
    """
    Trains a logistic regression model by splitting the data into
    non-overlapping groups and saves the model

    :param pd.DataFrame data_frame: data_frame to upsample
    :param str path_to_save: path to logistic regression model
    :return None
    """
    gene_id = data_frame['gene_id']
    X_data = data_frame.drop('label', axis=1)
    y_data = data_frame['label']

    gkf = GroupKFold(n_splits=3)
    train, test = next(gkf.split(X_data, y_data, groups=gene_id))

    X_data = X_data.drop('gene_id', axis=1)
    X_train = X_data.iloc[train, :]
    y_train = y_data.iloc[train]
    X_test = X_data.iloc[test]
    y_test = y_data.iloc[test]

    logreg_model = LogisticRegression(random_state=0)
    logreg_model.fit(X_train, y_train)
    y_pred = logreg_model.predict(X_test)
    accuracy = logreg_model.score(X_test, y_test)

    print(
        f'Accuracy of logistic regression classifier on test set: {accuracy}')
    print(classification_report(y_test, y_pred))
    pickle.dump(logreg_model, open(path_to_save, 'wb'))
    print(f'Model saved at {path_to_save}')

    y_pred_proba = logreg_model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    # Plot ROC curve
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train prediction model')

    parser.add_argument('--data', required=True,
                        default='',
                        metavar='path for data json file',
                        help='Json data file')

    parser.add_argument('--label', required=True,
                        default='',
                        metavar='path for data info file',
                        help='Info data file')

    parser.add_argument('--model_dir', required=True,
                        default='',
                        metavar='path for to save model',
                        help='Path for to save model')

    parser.add_argument('--standardize', required=False,
                        default=False,
                        metavar='path for data info file',
                        help='Info data file')

    args = parser.parse_args()
    JSON_DATA_PATH = args.data
    JSON_INFO_PATH = args.label
    PATH_SAVE_MODEL = args.model_dir
    STANDARDIZE_DATA = args.standardize

    """
    Start of data pre-processing
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    JSON_DATA_PATH = f'{dir_path}{JSON_DATA_PATH}'
    print(f'Training data from {JSON_DATA_PATH}')
    json_data_dataframe = load_data_json_file(JSON_DATA_PATH)

    JSON_INFO_PATH = f'{dir_path}{JSON_INFO_PATH}'
    print(f'Label data from {JSON_INFO_PATH}')
    data_info_dataframe = load_info_json_file(JSON_INFO_PATH)

    print('Processing data')
    concat_dataframe = concat_two_dataframe(
        data_info_dataframe, json_data_dataframe)

    data_frame_with_columns_dropped = drop_columns_from_dataframe(
        concat_dataframe, ['transcript_id', 'transcript_position'])

    one_hot_encoded_dataframe = one_hot_encode_nucleotide_dataframe(
        data_frame_with_columns_dropped
    )

    # Upsample datafram to have the same sample size for both label 0 and 1
    upsampled_dataframe = upsample_dataframe(one_hot_encoded_dataframe)

    if STANDARDIZE_DATA:
        subset_data = upsampled_dataframe[['dwelling_time_before', 'sd_before', 'mean_before',
                                           'dwelling_time_current', 'sd_current', 'mean_current',
                                           'dwelling_time_after', 'sd_after', 'mean_after']]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(subset_data)
        scaled_data = pd.DataFrame(scaled_data, 
            columns=[
            'dwelling_time_before', 'sd_before', 'mean_before',
            'dwelling_time_current', 'sd_current', 'mean_current',
            'dwelling_time_after', 'sd_after', 'mean_after'])
        upsampled_dataframe[
            ['dwelling_time_before', 'sd_before', 'mean_before',
            'dwelling_time_current', 'sd_current', 'mean_current',
            'dwelling_time_after', 'sd_after', 'mean_after']] = scaled_data

    print('Done processing data')
    """
    End of data pre-processing
    """
    PATH_SAVE_MODEL = f'{dir_path}{PATH_SAVE_MODEL}/model.sav'
    train_save_model(upsampled_dataframe, PATH_SAVE_MODEL)
