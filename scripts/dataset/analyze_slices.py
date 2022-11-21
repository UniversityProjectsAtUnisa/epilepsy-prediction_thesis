import json
import config
import pandas as pd
from typing import Any


def load_slices_metadata(output_path) -> dict[str, dict[str, dict[str, Any]]]:
    with open(output_path.joinpath(config.SLICES_FILENAME)) as f:
        return json.load(f)


def main():
    output_path = config.OUTPUT_PATH.joinpath(str(config.PREICTAL_SECONDS))
    slices_metadata = load_slices_metadata(output_path)

    data = {}

    for patient, patient_slices in slices_metadata.items():
        data[patient] = {}
        files_without_seizures = 0
        files_with_1_seizure = 0
        total_files = len(patient_slices.keys())
        training_seconds = 0
        test_negative_seconds = 0
        test_positive_seconds = 0
        test_negative_seconds_1 = 0
        test_positive_seconds_1 = 0
        total_seizures = 0
        for edf_filename, content in patient_slices.items():
            if edf_filename in config.DISCARDED_EDFS:
                continue
            n_seizures = content["n_seizures"]
            slices = content["slices"]
            if n_seizures == 0 and slices:
                files_without_seizures += 1
                for start, end, contains_seizure in slices:
                    assert not contains_seizure
                    training_seconds += end
            else:
                for start, end, contains_seizure in slices:
                    if not contains_seizure:
                        test_negative_seconds += end - start
                    else:
                        total_seizures += 1
                        total = end - start
                        assert total >= config.PREICTAL_SECONDS
                        negative = total - config.PREICTAL_SECONDS
                        positive = config.PREICTAL_SECONDS
                        test_positive_seconds += positive
                        test_negative_seconds += negative
                        if n_seizures == 1:
                            files_with_1_seizure += 1
                            test_positive_seconds_1 += positive
                            test_negative_seconds_1 += negative
        data[patient]["files_without_seizures"] = files_without_seizures
        data[patient]["total_files"] = total_files
        data[patient]["training_hours"] = training_seconds/3600
        data[patient]["n_seizures"] = total_seizures
        data[patient]["test_negative_hours"] = test_negative_seconds/3600
        data[patient]["test_positive_hours"] = test_positive_seconds/3600
        data[patient]["test_unbalance"] = test_positive_seconds/test_negative_seconds*100
        data[patient]["files_with_1_seizure"] = files_with_1_seizure
        data[patient]["test_negative_hours_1"] = test_negative_seconds_1/3600
        data[patient]["test_positive_hours_1"] = test_positive_seconds_1/3600
        data[patient]["test_unbalance_1"] = test_positive_seconds_1/test_negative_seconds_1*100

    df = pd.DataFrame(data).T
    df["files_without_seizures"] = pd.to_numeric(df["files_without_seizures"], downcast="integer")
    df["files_with_1_seizure"] = pd.to_numeric(df["files_with_1_seizure"], downcast="integer")
    df["total_files"] = pd.to_numeric(df["total_files"], downcast="integer")
    df["n_seizures"] = pd.to_numeric(df["n_seizures"], downcast="integer")
    print(df.sort_index().round(2))
    df.sort_index().round(2).to_csv(output_path.joinpath("slices_analysis.csv"))
    # print(f"{patient}: {files_without_seizures}/{total_files}, {training_seconds/3600:.2f} hours")


if __name__ == "__main__":
    main()
