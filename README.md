# Surface finish classifier

Classifies a two-point resistance log as a rough or smooth surface.

## Result

Leave-one-specimen-out accuracy on the 18 usable labelled recordings:
83.3% on real data only, 88.9% with synthetic padding (p = 0.0013).
A specimen's rough and smooth recordings are always held out together,
so the model never sees the test specimen in any form.

Cross-campaign check: trained only on the October S-series (1e6-1e9 ohm),
it got 4 of 4 usable August T-series recordings right (1e3-1e5 ohm).

## Requirements

    python3 -m pip install -r requirements.txt     # numpy, scikit-learn, matplotlib
    node                                           # only for the two JS checks

## Running it

Run everything from this folder, not from inside src/.

    python3 src/train_final.py     # retrain, validate, write out/model.json  (~40 s)
    python3 src/evaluate.py        # leave-one-specimen-out across model families
    python3 src/eval_synth.py      # does synthetic padding help?
    python3 src/loader.py          # data audit: lengths, overflow, ranges
    python3 web/build.py           # rebuild out/index.html from the trained model
    node web/verify.js             # JS engine vs python on all 32 files

## Using the website offline

Double-click `out/surface_classifier_local.html`. It opens in any browser,
needs no server and no internet, and nothing you upload leaves your machine.

Use that file, not `out/index.html`. They contain the same model and the same
code, but index.html is built for a host that supplies the HTML skeleton, so
on its own a browser renders it in quirks mode. The local file carries its own
skeleton.

Without internet the page falls back from IBM Plex to system fonts. Everything
else works unchanged.

## File map

    src/loader.py       parse the logs; 9.90E+37 becomes a missing reading
    src/segment.py      locate the loaded scan (onset to end of file)
    src/features.py     the 16 scale-free features of log10 resistance
    src/pipeline.py     feature set, crop augmentation, prediction crops
    src/synth.py        generator fitted to the real recordings
    src/evaluate.py     leave-one-specimen-out harness
    src/train_final.py  fit the shipped model and export it
    src/sweep.py        the 84-configuration search (kept for the write-up)
    src/nested.py       nested CV that exposed the sweep as selection bias
    web/engine.js       the pipeline ported to JavaScript
    web/verify.js       cross-check: both sides agree to 2.6e-14
    web/build.py        assemble the single-file page
    data/               all 32 CSV recordings

`src/dataset.py`, `src/windows.py` and `src/eval_models.py` are from the
earlier fixed-window approach that scored near chance. They are kept so the
write-up can show what was tried and rejected.

## Changing things

Retrain after editing anything in src/, then rebuild the page:

    python3 src/train_final.py && python3 web/build.py && node web/verify.js

Key settings are at the top of src/train_final.py: MAX_OVF (the 25% contact
loss quality rule), N_CROPS, N_SYN and W_SYN (synthetic padding).
