"""
test.py serves as a script to test the logistic model on unseen test data
"""
import argparse
import os
import pickle
import pandas as pd
from utils import load_data_json_file
from utils import one_hot_encode_nucleotide_dataframe


def predict_model_results(path_model: str, path_to_test_file: str,  path_save_results: str) -> None:
    """
    :param str path_model: path to saved model
    :param str path_to_test_file: path to test file
    :param str path_save_results: path to save results
    : return None
    """

    # Load model
    print(f'Load model from {PATH_MODEl}')
    logreg_model = pickle.load(open(path_model, 'rb'))

    # Load dataset and preprocess columns
    data_frame = load_data_json_file(path_to_test_file)
    data_frame_test = data_frame.drop(
        ['transcript_id', 'transcript_position'], axis=1)
    data_frame_test = one_hot_encode_nucleotide_dataframe(
        data_frame_test)

    # Predict score and save results
    predicted_prob = logreg_model.predict_proba(data_frame_test)
    df_prob = pd.DataFrame(predicted_prob)[1]
    df_prob = df_prob.to_frame(name='score')

    df_final = pd.concat([
        data_frame[['transcript_id', 'transcript_position']],
        df_prob],
        axis=1,
        join="inner")
    print(f'Save results to {path_save_results}')
    df_final.to_csv(path_save_results, index=False)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description='Test model on new data')
    parser.add_argument('--data', required=True,
                        default='',
                        metavar='path for data json file',
                        help='Json data file')

    parser.add_argument('--model_dir', required=True,
                        default='',
                        metavar='path for to save model',
                        help='Path for to save model')

    parser.add_argument('--save', required=True,
                        default='',
                        metavar='path to save results',
                        help='Path to save results')

    args = parser.parse_args()
    TEST_DATA_PATH = args.data
    PATH_MODEL = args.model_dir
    PATH_SAVE_RESULTS = args.save

    PATH_MODEl = f'{dir_path}{PATH_MODEL}'
    PATH_TEST_FILE = f'{dir_path}{TEST_DATA_PATH}'
    print(f'Test data from {PATH_TEST_FILE}')

    PATH_SAVE_RESULTS = f'{dir_path}{PATH_SAVE_RESULTS}/results.csv'
    predict_model_results(PATH_MODEl, PATH_TEST_FILE, PATH_SAVE_RESULTS)
