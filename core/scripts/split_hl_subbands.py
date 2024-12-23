import sys
import shutil
import os
import torch
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

sys.path.append(str(Path(__file__).parent.parent))
# from datasets_manager import get_datasets
from models.PQMF import PQMF

if __name__ == "__main__":
    opt = PQMF(2)
    # tr, cv, tt = get_datasets("AECChallenge")
    tr = "/home/deepni/trunk/gene-AEC-train-100-30-p06"
    cv = "/home/deepni/trunk/gene-AEC-test-4-1"
    tt = "/home/deepni/trunk/aec_test_set"

    out_tr = (
        "/home/deepni/trunk/gene-AEC-train-100-30-p06",
        "/home/deepni/trunk/gene-AEC-train-low-8k",
    )
    out_cv = ("/home/deepni/trunk/gene-AEC-test-4-1", "/home/deepni/trunk/gene-AEC-test-low-8k")
    out_tt = ("/home/deepni/trunk/aec_test_set", "/home/deepni/trunk/aec_test_set-low-8k")

    # for dmic, dref, dsph, fname in tr:
    #     print(dmic.shape, dref.shape, dsph.shape, fname)
    #     sys.exit()

    for dirn, pat in zip([tr, cv, tt], [out_tr, out_cv, out_tt]):
        for f in tqdm(list(map(str, Path(dirn).rglob("*.wav")))):
            data, fs = sf.read(f)
            data = torch.from_numpy(data)[None, :].float()  # B,T
            outf = f.replace(*pat)

            if not os.path.exists(os.path.dirname(outf)):
                os.makedirs(os.path.dirname(outf))

            out = opt.analysis(data)
            sf.write(outf, out[0], fs // 2)
