"""
utils.py serves as a utility module for DSA4262
"""
from ast import List
import json
import pandas as pd
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
                        current_row_transcript_position_features_list.append(
                            combined_nucleotides)
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
            'transcript_id', 'transcript_position', 'nucleotides',
            'dwelling_time_before', 'sd_before', 'mean_before',
            'dwelling_time_current', 'sd_current', 'mean_current',
            'dwelling_time_after', 'sd_after', 'mean_after']
        )

        nucleotide_column = process_nucleotide_column(df_from_json)
        df_from_json = df_from_json.drop('nucleotides', axis=1)
        df_from_json = pd.concat([df_from_json, nucleotide_column], axis=1)
        df_from_json['transcript_position'] = df_from_json['transcript_position'].astype(
            object)

        return df_from_json


def process_nucleotide_column(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas dataframe after processing the nucleotides column.

    :param data_frame pd.DataFrame
    :return pandas dataframe
    """
    nucleotides_column = data_frame['nucleotides']
    nucleotides_previous = nucleotides_column.str[:5]
    nucleotides_current = nucleotides_column.str[1:6]
    nucleotides_next = nucleotides_column.str[2:]
    processed_nucleotides = pd.concat([
        nucleotides_previous,
        nucleotides_current,
        nucleotides_next],
        axis=1)
    processed_nucleotides = pd.DataFrame(processed_nucleotides)
    processed_nucleotides.columns = [
        'nucleotide_before',
        'nucleotide_current',
        'nucleotide_after'
    ]
    return processed_nucleotides


def load_info_json_file(file_path: str) -> pd.DataFrame:
    """
    Returns a pandas dataframe loaded from a data.info file

    :param str path: path to json file
    :return pandas dataframe
    """
    dataframe = pd.read_csv(file_path, header=0)
    dataframe['transcript_position'] = dataframe['transcript_position'].astype(
        object)
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
    processed_dataframe = processed_dataframe.loc[:,
                                                  ~processed_dataframe.columns.duplicated()]
    return processed_dataframe


def drop_columns_from_dataframe(df_1: pd.DataFrame, columns: List(str)) -> pd.DataFrame:
    """
    Returns a pandas dataframe loaded by concating two dataframes
    and removing the first two dataframe.

    :param df_1 pd.DataFrame
    :param List(str) columns: List of columns to drop
    :return pandas dataframe
    """
    dataframe_with_columns_drop = df_1.drop(columns, axis=1)
    return dataframe_with_columns_drop


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


def one_hot_encode_nucleotide_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with nucleotides one-hot encoded.

    :param pd.DataFrame data_frame: data_frame to one-hot encode nucleotide
    :return pandas dataframe
    """
    # Retrieve nucleotides_before, nucleotide_current, nucleotide_after
    # for one-hot coding
    nucleotides = data_frame[[
        'nucleotide_before',
        'nucleotide_current',
        'nucleotide_after'
    ]]

    # One-hot encode nucleotide
    nucleotides = pd.get_dummies(nucleotides)

    # Removes nucleotides_before, nucleotide_current, nucleotide_after
    data_frame = drop_columns_from_dataframe(
        data_frame, [
            'nucleotide_before',
            'nucleotide_current',
            'nucleotide_after'
        ])

    # Concat dataframe with nucleotides dropped and
    # one-hot encoded
    concat_dataframe = concat_two_dataframe(
        data_frame, nucleotides)

    return concat_dataframe
