from pathlib import Path
import config as cfg
from helper_functions import record_movements


if __name__ == "__main__":
    folder = Path(cfg.RECORDINGS_FOLDER)
    n_csv = sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".csv")
    file_id = n_csv
    record_movements(file_id)
