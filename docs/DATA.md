# Data sources, citations and licences

All three databases are published on PhysioNet. On 2026-11-27 each database's PhysioNet page (version 1.0.0) stated the
**Open Data Commons Attribution License v1.0** (ODC-By 1.0, https://opendatacommons.org/licenses/by/1-0/). That licence
requires attribution and that notices are kept. It covers the data, not this project's code.

| Database | Used here for | In this repository |
|---|---|---|
| MIT-BIH Arrhythmia Database | The app's model; study training, validation and the DS2 test | All 48 records (`results/mitdb/`), copies of records 100, 106 and 228 (`sample_data/`), and data derived from it (demo windows, XAI examples, report files, the trained model) |
| MIT-BIH Supraventricular Arrhythmia Database (SVDB) | Study training and validation | No signal or annotation files. Counts and scores in report files only |
| St Petersburg INCART 12-lead Arrhythmia Database | Study training, validation and a test on held-out patients | No signal or annotation files. Counts and scores in report files only |

The sealed study test files and per-beat prediction files are not in the repository.

## Citations

**MIT-BIH Arrhythmia Database**, https://physionet.org/content/mitdb/1.0.0/

- Moody, G., & Mark, R. (2005). MIT-BIH Arrhythmia Database (version 1.0.0). PhysioNet. RRID:SCR_007345.
  https://doi.org/10.13026/C2F305
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50
  (May-June 2001). (PMID: 11446209)

**MIT-BIH Supraventricular Arrhythmia Database**, https://physionet.org/content/svdb/1.0.0/, https://doi.org/10.13026/C2V30W

- Greenwald SD. Improved detection and classification of arrhythmias in noise-corrupted electrocardiograms using
  contextual information. Ph.D. thesis, Harvard-MIT Division of Health Sciences and Technology, 1990.
  https://dspace.mit.edu/handle/1721.1/29206

**St Petersburg INCART 12-lead Arrhythmia Database**, https://physionet.org/content/incartdb/1.0.0/,
https://doi.org/10.13026/C2V88N

- The page asked only for the standard PhysioNet citation; it listed no separate original publication.

**PhysioNet (standard citation, as the three pages gave it)**

- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026).
  PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z

Check the pages again before publication: citation requests can change.
