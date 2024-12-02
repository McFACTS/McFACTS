import os.path
import pickle
from typing import Any


def pickle_state(object_to_save: Any, folder_name: str, file_name: str, overwrite: bool = False):
    path_exists = os.path.isdir(folder_name)

    if not path_exists:
        os.makedirs(folder_name, exist_ok=True)

    full_path = os.path.join(folder_name, file_name)
    file_exists = os.path.isfile(full_path)

    if not overwrite and file_exists:
        raise FileExistsError(f"The file at {full_path} already exists.")

    with open(full_path, "wb") as out:
        pickle.dump(object_to_save, out, pickle.HIGHEST_PROTOCOL)
