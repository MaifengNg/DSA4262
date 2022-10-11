"""
utils.py serves as a utility module for DSA4262
"""
import json
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample


def load_data_json_file(file_path: str) -> pd.DataFrame:
    """
    Returns a pandas dataframe loaded from a data.json file

    :param str path: path to json file
    :return pandas dataframe
    """
    json_nested_lists = []
    with open(file_path, 'r', encoding="utf-8") as file_reader:
        all_lines_from_file_reader = file_reader.readlines()
        for line in all_lines_from_file_reader:
            json_from_line = json.loads(line)

            current_row_transcript_position_features_list = []
            for transcript in json_from_line:
                json_transcript = json_from_line[transcript]
                current_row_transcript_position_features_list.append(
                    transcript)

                for position_within_transcript in json_transcript:
                    current_row_transcript_position_features_list.append(
                        position_within_transcript)
                    json_position_within_transcript = json_transcript[position_within_transcript]

                    for combined_nucleotides in json_position_within_transcript:
                        nested_list_of_features = json_position_within_transcript[
                            combined_nucleotides]
                        mean_list_of_features = [0] * 9
                        num_list_of_features = len(nested_list_of_features)

                        for i in range(num_list_of_features):
                            current_list_of_features = nested_list_of_features[i]

                            for j in range(9):
                                mean_list_of_features[j] += current_list_of_features[j]

                        mean_list_of_features = [
                            feature / num_list_of_features for feature in mean_list_of_features
                        ]
                        current_row_transcript_position_features_list.extend(
                            mean_list_of_features)
            json_nested_lists.append(
                current_row_transcript_position_features_list)

        df_from_json = pd.DataFrame(json_nested_lists, columns=[
            'transcript_id', 'transcript_position',
            'dwelling_time_before', 'sd_before', 'mean_before',
            'dwelling_time_current', 'sd_current', 'mean_current',
            'dwelling_time_after', 'sd_after', 'mean_after']
        )

        return df_from_json


def load_info_json_file(file_path: str) -> pd.DataFrame:
    """
    Returns a pandas dataframe loaded from a data.info file

    :param str path: path to json file
    :return pandas dataframe
    """
    dataframe = pd.read_csv(file_path, header=0)
    dataframe = dataframe.drop('gene_id', axis=1)
    return dataframe


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


def predict_model_results(path_model: str, path_to_test_file: str,  path_save_results: str) -> None:
    """
    :param str path_model: path to saved model
    :param str path_to_test_file: path to test file
    :param str path_save_results: path to save results
    : return None
    """
    logreg_model = pickle.load(open(path_model, 'rb'))
    data_frame = load_data_json_file(path_to_test_file)
    data_frame_test = data_frame.drop(
        ['transcript_id', 'transcript_position'], axis=1)

    predicted_prob = logreg_model.predict_proba(data_frame_test)
    df_prob = pd.DataFrame(predicted_prob)[0]
    df_prob.rename(columns={0: "score"})

    df_final = pd.concat([
        data_frame[['transcript_id', 'transcript_position']],
        df_prob],
        axis=1,
        join="inner")
    df_final.to_csv(path_save_results, index=False)
