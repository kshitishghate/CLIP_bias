import os

import pandas as pd

global prev
prev = None

def test_already_run(model_name, test, file_name, hard_reload=False):
    if not os.path.exists(file_name):
        return False
    global prev
    if prev == None or hard_reload:
        previous_results = pd.read_csv(file_name)
    else:
        previous_results = prev
    relevant_results = previous_results[
        (previous_results['model'] == model_name)
        & (True if 'X' not in test.index else (previous_results['X'] == test['X']))
        & (True if 'Y' not in test.index else (previous_results['Y'] == test['Y']))
        & (True if 'A' not in test.index else (previous_results['A'] == test['A']))
        & (True if 'B' not in test.index else (previous_results['B'] == test['B']))
        & (True if 'Image Test' not in test.index else (previous_results['Image Test'] == test['Image Test']))
        & (True if 'Text Test' not in test.index else (previous_results['Text Test'] == test['Text Test']))
        & (True if 'Target' not in test.index else (previous_results['Target'] == test['Target']))
        & (True if 'na' not in test.index else (previous_results['na'] == test['na']))
        & (True if 'nt' not in test.index else (previous_results['nt'] == test['nt']))
        & (True if 'naa' not in test.index else (previous_results['naa'] == test['naa']))
        & (True if 'nab' not in test.index else (previous_results['nab'] == test['nab']))
        & (True if 'context' not in test.index else (previous_results['context'] == test['context']))
        & (True if 'order' not in test.index else (previous_results['order'] == test['order']))
        & (True if 'category' not in test.index else (previous_results['category'] == test['category']))
    ]
    return len(relevant_results) > 0


def save_test_results(result, file_name):
    if type(result) == pd.Series:
        result = pd.DataFrame(result).T
    if os.path.exists(file_name):
        # Read file header to make sure we can get the same column order
        header = pd.read_csv(file_name, nrows=0).columns.tolist()

        # If there are new columns, add them to the previous results
        new_columns = [c for c in result.columns if c not in header]
        if len(new_columns) > 0:
            previous_results = pd.read_csv(file_name)
            all_results = pd.concat([previous_results, result]).reset_index(drop=True)
            all_results.to_csv(file_name, index=False)
        else:
            result[header].to_csv(file_name, index=False, mode='a', header=False)
    else:
        result.to_csv(file_name, index=False)

