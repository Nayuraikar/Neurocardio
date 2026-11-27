Sample ECGs: MIT-BIH Arrhythmia Database (PhysioNet, ODC-By 1.0). Cite it when shown.
All six records are from the DS2 test list, so the deployed model did not train on them.
Keep these OUT of the git repo.

Record  Use it to show
100     Mostly normal rhythm with 33 S beats. A good first demo.
212     Almost all N. Shows a clean, quiet result.
221     A record with V beats, for the ventricular-class demo and playback.
233     V beats plus some F, for the XAI and uncertainty pages.
232     1,382 S beats, which the model mostly misses (74 caught in the study). Shows the S weakness honestly.
213     361 F beats, which the study model essentially missed. Shows the F limitation.

Avoid training records (for example 208, 119, 203) for performance claims, and skip 202,
which was used for the deployed model's early stopping.
