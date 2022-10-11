"""
test.py serves as a utility module for DSA4262
"""
import os
import pickle
import pandas as pd
from utils import load_data_json_file


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
    df_prob = df_prob.to_frame(name='score')

    df_final = pd.concat([
        data_frame[['transcript_id', 'transcript_position']],
        df_prob],
        axis=1,
        join="inner")
    df_final.to_csv(path_save_results, index=False)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    PATH_MODEl = f'{dir_path}/Data/model.sav'
    PATH_TEST_FILE = f'{dir_path}/Data/data.json'
    PATH_SAVE_RESULTS = f'{dir_path}/Data/results.csv'
    predict_model_results(PATH_MODEl, PATH_TEST_FILE, PATH_SAVE_RESULTS)
