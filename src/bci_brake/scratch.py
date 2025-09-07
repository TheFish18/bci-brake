import json
from pathlib import Path

import mat73

from bci_brake.constants import dist_data_dir

x_feats = ['EOGv', 'Fp1', 'Fp2', 'AF3', 'AF4', 'EOGh', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'EMGf', 'lead_gas', 'lead_brake', 'dist_to_lead', 'wheel_X', 'wheel_Y', 'gas', 'brake']

mat_dir = Path("/home/josh/projects/python/BCIBrake/data/mats")

for mat_p in mat_dir.glob("*.mat"):
    data = mat73.loadmat(mat_p)
    x_feats_ = data["cnt"]["clab"]

    if x_feats != x_feats_:
        print(mat_p.stem)

# with dist_data_dir.joinpath("x_features.json").open("w") as f:
#     json.dump(x_feats, f, indent=4)
