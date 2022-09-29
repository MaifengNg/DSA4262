"""
utils.py serves as a utility module for DSA4262
"""
import json
import os
import pandas as pd


def load_json_file(file_path: str) -> pd.DataFrame:
    """
    Returns a pandas dataframe loaded from a json file

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
            'Transcript', 'Position',
            'dwelling_time_before', 'sd_before', 'mean_before',
            'dwelling_time_current', 'sd_current', 'mean_current',
            'dwelling_time_after', 'sd_after', 'mean_after']
        )

        return df_from_json


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    JSON_PATH = f'{dir_path}/Data/data.json'
    json_dataframe = load_json_file(JSON_PATH)
    print(json_dataframe.head())
