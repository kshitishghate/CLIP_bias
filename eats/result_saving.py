import os

import pandas as pd


def test_already_run(model_name, test, file_name):
    if not os.path.exists(file_name):
        return False
    previous_results = pd.read_csv(file_name)
    relevant_results = previous_results[
        (previous_results['model'] == model_name)
        & (previous_results['X'] == test['X'])
        & (previous_results['Y'] == test['Y'])
        & (previous_results['A'] == test['A'])
        & (previous_results['B'] == test['B'])
        & (previous_results['Title'] == test['Title'])
        & (previous_results['na'] == test['na'])
        & (previous_results['nt'] == test['nt'])
        & (True if 'context' not in test.index else (previous_results['context'] == test['context']))
    ]
    return len(relevant_results) > 0


def save_test_results(result, file_name):
    new_result = pd.DataFrame(result).T
    if os.path.exists(file_name):
        previous_results = pd.read_csv(file_name)
        all_results = pd.concat([previous_results, new_result]).reset_index(drop=True)
    else:
        all_results = new_result

    all_results.to_csv(file_name, index=False)
