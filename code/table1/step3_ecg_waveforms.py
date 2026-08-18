#!/usr/bin/env python3
"""
Fetch the PTB Diagnostic ECG Database from PhysioNet and build the waveform
tensor Table 1's ECG row is computed on.

Records are pulled one at a time over wfdb's byte-range reader with
sampto=10,000, so each record costs ~300 KB instead of its full ~2.5 MB. Only
the two diagnoses the task uses are kept — myocardial infarction and healthy
control, read out of the header comments — and everything else in the 549
records is dropped. Signals are the first 12 of the 15 channels, NaN/Inf
zeroed, then polyphase-resampled 1000 Hz -> 100 Hz to 1,000 steps.

Patient identity is saved alongside the waveforms because the split in step6 is
at the patient level, not the recording level: several patients contribute more
than one recording, and splitting on recordings would leak.

    python3 step3_ecg_waveforms.py [--out-dir DIR]

Inputs   https://physionet.org/files/ptbdb/1.0.0/ (public, credential-free)
Outputs  data/table1/ecg/ptb_waveforms.npy    (N, 1000, 12) float32
         data/table1/ecg/ptb_labels.npy       (N,)          int32  0=HC 1=MI
         data/table1/ecg/ptb_patient_ids.npy  (N,)          str
         data/table1/ecg/RECORDS              cached record list
Needs    wfdb, scipy. Expect 30-60 min on the first run; the numpy cache is
         reused afterwards.
"""
import _common as C

import argparse
import time
import urllib.request

import numpy as np

PTBDB_BASE = "https://physionet.org/files/ptbdb/1.0.0/"
PTBDB_PNN = "ptbdb/1.0.0"

ORIG_FS = 1000       # PhysioNet sampling rate (Hz)
TARGET_FS = 100      # resample target (Hz)
DURATION_S = 10      # seconds kept per recording
ORIG_LEN = ORIG_FS * DURATION_S     # 10,000 samples
SEQ_LEN = TARGET_FS * DURATION_S    # 1,000 steps
N_LEADS = 12         # standard 12 of the 15 PTB channels


def get_record_list(data_dir):
    cache = data_dir / "RECORDS"
    if cache.exists():
        return [l.strip() for l in cache.read_text().splitlines() if l.strip()]
    url = PTBDB_BASE + "RECORDS"
    print(f"  fetching record list from {url}", flush=True)
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    cache.write_text(text)
    return [l.strip() for l in text.splitlines() if l.strip()]


def parse_diagnosis(comments):
    for c in comments or []:
        c = c.strip()
        if c.startswith("Reason for admission:"):
            return c.split(":", 1)[1].strip().lower()
        if c.startswith("Diagnosis:"):
            return c.split(":", 1)[1].strip().lower()
    return None


def load_ptb(data_dir):
    import wfdb
    from scipy.signal import resample_poly

    cache_wav = data_dir / "ptb_waveforms.npy"
    cache_lbl = data_dir / "ptb_labels.npy"
    cache_pid = data_dir / "ptb_patient_ids.npy"
    if cache_wav.exists() and cache_lbl.exists() and cache_pid.exists():
        print("  reusing numpy cache", flush=True)
        return (np.load(cache_wav), np.load(cache_lbl),
                np.load(cache_pid, allow_pickle=True))

    records = get_record_list(data_dir)
    print(f"  {len(records)} records listed; streaming first {DURATION_S}s of each",
          flush=True)

    waveforms, labels, patient_ids = [], [], []
    skipped = 0
    t0 = time.perf_counter()

    for i, rec_path in enumerate(records):
        patient_id = rec_path.split("/")[0]      # "patient001"
        rec_name = rec_path.split("/")[-1]       # "s0010_re"
        pn_dir = f"{PTBDB_PNN}/{patient_id}"

        try:
            hdr = wfdb.rdheader(rec_name, pn_dir=pn_dir)
        except Exception:
            skipped += 1
            continue

        diagnosis = parse_diagnosis(hdr.comments)
        if diagnosis is None:
            continue
        if "myocardial infarction" in diagnosis:
            label = 1
        elif "healthy control" in diagnosis:
            label = 0
        else:
            continue                              # neither class — not part of the task

        try:
            sig, _ = wfdb.rdsamp(rec_name, pn_dir=pn_dir, sampto=ORIG_LEN)
        except Exception:
            skipped += 1
            continue

        if sig.shape[0] < ORIG_LEN:               # short recording — zero-pad
            pad = np.zeros((ORIG_LEN - sig.shape[0], sig.shape[1]), dtype=np.float32)
            sig = np.vstack([sig, pad])
        sig = sig[:ORIG_LEN, :N_LEADS].astype(np.float32)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        sig_ds = resample_poly(sig, 1, ORIG_FS // TARGET_FS, axis=0).astype(np.float32)
        if sig_ds.shape[0] > SEQ_LEN:
            sig_ds = sig_ds[:SEQ_LEN]
        elif sig_ds.shape[0] < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - sig_ds.shape[0], N_LEADS), dtype=np.float32)
            sig_ds = np.vstack([sig_ds, pad])

        waveforms.append(sig_ds)
        labels.append(label)
        patient_ids.append(patient_id)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(records)} records "
                  f"({len(waveforms)} kept, {time.perf_counter()-t0:.0f}s)", flush=True)

    waveforms = np.array(waveforms, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    patient_ids = np.array(patient_ids)
    n_mi, n_hc = int((labels == 1).sum()), int((labels == 0).sum())
    print(f"  kept {len(waveforms)} recordings (MI={n_mi} HC={n_hc}), "
          f"skipped {skipped}, shape={waveforms.shape}", flush=True)

    np.save(cache_wav, waveforms)
    np.save(cache_lbl, labels)
    np.save(cache_pid, patient_ids)
    return waveforms, labels, patient_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(C.DATA / "ecg"))
    args = ap.parse_args()
    out = C.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    load_ptb(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
