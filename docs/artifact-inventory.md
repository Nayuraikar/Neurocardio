# Kaggle artifact inventory

Inspected recursively: `/Users/nayana/Downloads/Neurocardio/results` (214 files). Original artifacts were read only; no files in `results/` were changed.

## Selection and verification findings

- The matching notebook is `/Users/nayana/Downloads/ecg-aiml (1).ipynb` (53 cells). Its model selection code chooses the highest validation macro-F1 among BiLSTM variants and exports `neurocardio_final.keras`. The output config says the chosen variant is `bilstm_aug_sqrtw` (CNN-BiLSTM + training-only augmentation, square-root weighted cross entropy).
- `neurocardio_final.keras` and `bilstm_aug_sqrtw.keras` have identical `config.json` and `model.weights.h5` SHA-256 values; only Keras save metadata differs. The final export is therefore the selected checkpoint. SHA-256 of the complete final file: `0e7083d56c31a4e43f6cd332b375b8480446485b60b80f52c3c7c5380508becc`.
- The exported model config identifies Keras 3.13.2, Functional `cnn_bilstm`, one input `ecg_input` shaped `(batch, 720, 1)`, one softmax output `probs` of four classes; class order is N, S, V, F. `conv3` and `logits` are named in the XAI config.
- Exact inference preprocessing in the exported config/notebook: 360 Hz; 720 samples (2 s), centered at the R peak (360 samples either side); apply a fourth-order Butterworth 0.5–40 Hz zero-phase SOS bandpass to the full continuous signal before segmentation; per-window z-score with epsilon `1e-8`; raw softmax argmax is the primary decision rule. Augmentation is training-only and is not an inference step.
- RR-related artifacts/models exist, and the notebook defines five RR features and their train-statistic scaling for experimental RR models. However, the selected final export explicitly has `rr_features: null` and one ECG input. The selected model does not require post-beat RR. This contradicts the request premise that the existing selected model uses post-beat RR; the project must follow the verified export for exact inference.
- Selective thresholds are present (`selective_thresholds.json`): temperature 1.3 and conservative/balanced/permissive operating points, validation-only. These are explicitly unstable/unproven on new patients due to sparse validation S/F examples; product policy should label them exploratory and configurable, with human review.
- Notebook evaluation uses annotated MIT-BIH beat positions. The config says detected R peaks are acceptable for deployment, but no deployed R-peak detector/configuration or notebook comparison on detector-generated beats is provided. This is a material missing preprocessing artifact for arbitrary uploaded recordings.
- Runtime verification could not be performed in this environment: TensorFlow/Keras and WFDB are not installed. Model config/weights can be inspected structurally, but load/inference and TensorFlow compatibility remain unverified until dependencies are installed. No architecture fallback should be used.
- Feature scalers: none required for selected ECG-only model. `class_scales.json` exists but explicitly says ablation-only and not applied. RR mean/std are embedded in preprocessing config only if an RR model had been selected; they are null for this final model.
- Notebook versions `ecg-aiml.ipynb`, `ecg-aiml (1).ipynb`, `ecg-aiml (2).ipynb`, and `ecg-aiml (3).ipynb` were found in Downloads and contain the same selected-model/export code family; `(1)` was used as the detailed cross-reference. No notebook file was found inside this project root.

## Artifact groups

- **Models (8):** `.keras` full checkpoints; `.keras.split` sidecar text files exist for nine experimental model names, but no sidecar for `neurocardio_final.keras`.
- **Configuration/calibration (9 JSON):** preprocessing, selective thresholds, class scales, split selection, CV state/results, test metrics/raw primary/ablation.
- **Evaluation/history (20 CSV):** training histories, final/model comparison, per-record and split audits, selective reports/operating points, class distributions, window-length audit.
- **Plots/XAI (19 PNG):** confusion matrices, PR curves, coverage-risk, training curves, class distribution, XAI examples, and saved notebook plot outputs.
- **NumPy (5):** train/validation/test record lists, CV state NPZ, test XAI examples NPZ.
- **Dataset (144 files):** 48 MIT-BIH records each with `.hea`, `.dat`, `.atr` under `results/mitdb/`.

## Complete file inventory

| File | Bytes | Type | SHA-256 |
|---|---:|---|---|
| `results/__results___files/__results___12_0.png` | 74501 | Plot, XAI, or notebook output image | `3623ed684d9149140cd4bd3dbdd1de97da942bd6b233cceb1d138f27e008f498` |
| `results/__results___files/__results___16_0.png` | 68729 | Plot, XAI, or notebook output image | `63a5e2c4486db475ba75483596925b86cd6fd253210f0360619c9981cd6850f8` |
| `results/__results___files/__results___17_1.png` | 15638 | Plot, XAI, or notebook output image | `184501b0e4138111827ded2600d78084288fd0892200fd4d6de771de8ca2c714` |
| `results/__results___files/__results___35_0.png` | 36428 | Plot, XAI, or notebook output image | `f560e10ed54460a07fff7ad4e2cd6d5dd65b34cb8bab630feef7cbd1719f2141` |
| `results/__results___files/__results___35_1.png` | 41126 | Plot, XAI, or notebook output image | `ec97d2a3194ae88831cc566a78dc84d146f3a0ad7dcfd5f04b400f8db1a32b96` |
| `results/__results___files/__results___36_0.png` | 42378 | Plot, XAI, or notebook output image | `f916e14e58d70a2b49892d87b9416f53cc164e7f6363b137753d64be2136f875` |
| `results/__results___files/__results___38_7.png` | 40792 | Plot, XAI, or notebook output image | `6d13710da18ecbf35ccd68eeb5df83c5ffa77dc396012bec327fd0c662aefc7f` |
| `results/__results___files/__results___41_0.png` | 325475 | Plot, XAI, or notebook output image | `1a585cf022e1157afa73595732a4c09105d106db7ab6641c9c38653e6db85264` |
| `results/__results___files/__results___48_6.png` | 38074 | Plot, XAI, or notebook output image | `a96e2129722ec5b1ad84b539c8fe433077f5af1ee2b1d914f28ecb1e5307971f` |
| `results/__results___files/__results___49_3.png` | 39923 | Plot, XAI, or notebook output image | `889152524eb6ea42fa61b935064ec9eefc739f6a940d9ff73ae6abd6075ed968` |
| `results/__results___files/__results___5_1.png` | 57104 | Plot, XAI, or notebook output image | `2ff2c16fec6554a65c91577316b724ab9082d49e2f28c704f2415d9c687655cb` |
| `results/bilstm_aug_sqrtw.keras` | 1794687 | Keras model | `56edc12cc2ca6d391657e45b896c1627b83fd825f431cb3bd33bd707f4bffab1` |
| `results/bilstm_aug_sqrtw.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/bilstm_ce.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/bilstm_focal_g1.0.keras` | 1794859 | Keras model | `f7fe1ef7737da485e05442c99a261d4705515dca7274aae8addcee4dbc31e949` |
| `results/bilstm_focal_g1.0.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/bilstm_focal_g2.0.keras` | 1794859 | Keras model | `8a5e2acaebdf8d8e6a8a84214db86b90fb468cbcae104c18a94d892b04113cfd` |
| `results/bilstm_focal_g2.0.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/bilstm_rr_aug_sqrtw.keras` | 1817868 | Keras model | `480f9ca28175bc93efa863f37a3273c6aac66e2c558e011e50aba59b59ee05cb` |
| `results/bilstm_rr_aug_sqrtw.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/bilstm_rr_sqrtw.keras` | 1817864 | Keras model | `1d0b558ea6aa4481b8cc5996356484f5bd367089b1132b5bc5d4e935ee5a76fb` |
| `results/bilstm_rr_sqrtw.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/bilstm_sqrtw.keras` | 1794687 | Keras model | `b73886815f03f3f2aa8bcde6efe10054d4c85142112b4fffdad29c5c69807308` |
| `results/bilstm_sqrtw.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/bilstm_wce.keras` | 1794675 | Keras model | `5e87b32631898610ec085b07646bcd5498cdf530ef82823c04877f064dbb1b6d` |
| `results/bilstm_wce.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/class_distribution.png` | 26452 | Plot, XAI, or notebook output image | `9b64ddd7f265dcf6fc59a1ccc7d5bbf5289dde3256e2d01959e3b8d7f1385d93` |
| `results/class_scales.json` | 215 | Configuration or metrics JSON | `95c39e98324f46ba8ab397320076e7067f15d7c2a56ad46140ea30116d532be5` |
| `results/cnn_ce.keras.split` | 16 | Model split metadata | `e30c4628fb250808e514be7f9498b1d579c9197deeadfc3041f85ecd2654ad44` |
| `results/confusion_matrix.png` | 60438 | Plot, XAI, or notebook output image | `495ec213baa1123fdc8fd4c9ef207edbae4b865c5897cc049a3a7a6306c9eea4` |
| `results/coverage_vs_risk.png` | 63846 | Plot, XAI, or notebook output image | `c732e167552bb8452d30ef872f5213cc79204e57b8749ac037efb312da02cabe` |
| `results/cv_F_record_audit.csv` | 1210 | Metrics/history/audit CSV | `49a76bd37028e10810fed041e4cf6c5704467128354f6ea3b2d029a3ac4d9082` |
| `results/cv_confusion_matrix.png` | 63640 | Plot, XAI, or notebook output image | `5efb8d3a47a0d9a79201e33b868763c8f15bbbecb9cf5fc5f220a60903012170` |
| `results/cv_results.json` | 11202 | Configuration or metrics JSON | `cc52eae0d7ef77e8c6308437e8e3ae1229a66376d1521a28fd456be902d3bf34` |
| `results/cv_state_bilstm_aug_sqrtw.json` | 1001 | Configuration or metrics JSON | `137767095f757e7a37df8306f5cd0c10f47f15247a2f371a3309cd494d141487` |
| `results/cv_state_bilstm_aug_sqrtw.npz` | 1710854 | NumPy records/state/examples | `afe038a1b60931a5c0effd9db82f1d28493e8f699983cdb12c9d8b7bb2d5855e` |
| `results/experiment_table_test.csv` | 597 | Metrics/history/audit CSV | `fafc002848c2b2bf43836a30887171befd205026a93323c221853154d5d526f1` |
| `results/final_summary.csv` | 530 | Metrics/history/audit CSV | `cf066e95086f6c8d0b5abdd8b21a1ae582cf8fa45cc160f2133f91d94711694f` |
| `results/hist_bilstm_aug_sqrtw.csv` | 2693 | Metrics/history/audit CSV | `d68882a0dac8e6c4e7e3c0bd956dda8e53edb31fb13867ed9162188f55e30ddc` |
| `results/hist_bilstm_focal_g1.0.csv` | 3863 | Metrics/history/audit CSV | `c4d211f923ce89d46adc6e2a029153357eb0fd3d40d8636f70b898783e2de484` |
| `results/hist_bilstm_focal_g2.0.csv` | 5982 | Metrics/history/audit CSV | `5e35d80d6f2fc9baf53bc72a76b4470b9275be71c6c7670526585813b649bf25` |
| `results/hist_bilstm_rr_aug_sqrtw.csv` | 2051 | Metrics/history/audit CSV | `e08c1842362b5017e7149907dc260bd83d0013f9b5d3595729d384f85be18d51` |
| `results/hist_bilstm_rr_sqrtw.csv` | 4624 | Metrics/history/audit CSV | `973b3d68cbee2fa2139d345c9e05a8670a209ce22f5055d8d768702346bfa552` |
| `results/hist_bilstm_sqrtw.csv` | 3431 | Metrics/history/audit CSV | `96c8596d0cb4cba9db2f24e72fadd86d51d1322938cd174bc91f92cc9426a0d7` |
| `results/hist_bilstm_wce.csv` | 3888 | Metrics/history/audit CSV | `7abc02b1fc098be37c5efc50dd5e423771bb03fe22e69849a027d7ec1f6f139c` |
| `results/minority_concentration_by_split.csv` | 295 | Metrics/history/audit CSV | `81da1628224dce09a87dc9d32fbd586a2f713debce9185b1dc88fd3d0a718f61` |
| `results/mitdb/100.atr` | 4558 | MIT-BIH WFDB data/annotation | `8d8a5349fb16638ebbf649f1779d12e96d91b736b2aafe59db43719ae583d471` |
| `results/mitdb/100.dat` | 1950000 | MIT-BIH WFDB data/annotation | `b2ea3c250e56e48f4b7b90697832b8ecd1afa1e0bb31f2dcfea4ed6e1075a639` |
| `results/mitdb/100.hea` | 143 | MIT-BIH WFDB data/annotation | `60ebc904c7bf3e04d142638d3fd5c903e8dc9f10f1ea3264e07926aa089ee75e` |
| `results/mitdb/101.atr` | 3768 | MIT-BIH WFDB data/annotation | `441cdd6486cfdf4c53e344d1048ab81296773e7058d53069022adc68543a0663` |
| `results/mitdb/101.dat` | 1950000 | MIT-BIH WFDB data/annotation | `698d1ea6f472d23ca50317c72c96cda2698badd8578220ed0380cdf241e39006` |
| `results/mitdb/101.hea` | 131 | MIT-BIH WFDB data/annotation | `d5f02fbe8673fa05465442191b98ca0d28a1670e7ef0e83fb9ef8723113a311c` |
| `results/mitdb/102.atr` | 4424 | MIT-BIH WFDB data/annotation | `24e29031c081806ff6d4ec20fb386e28d8f885f0dca5f74dc29c881e8b196d09` |
| `results/mitdb/102.dat` | 1950000 | MIT-BIH WFDB data/annotation | `2b56039f9ece83b53511e0fc24e95f23b90712c8017fe4d915e040e78713c38e` |
| `results/mitdb/102.hea` | 204 | MIT-BIH WFDB data/annotation | `a22f09934ee1cfe339d1502f68eb3f0175815f47273b81ea55f75aeb5584cb0d` |
| `results/mitdb/103.atr` | 4196 | MIT-BIH WFDB data/annotation | `392ce37fbc7e89b3f14998e197a844f23cce6a3db6edaa71b4b49bf3ca5b6596` |
| `results/mitdb/103.dat` | 1950000 | MIT-BIH WFDB data/annotation | `f7994f0bd196730779998b4bddf26cb47f396599fb370a08b5cb29a3ac4d555d` |
| `results/mitdb/103.hea` | 142 | MIT-BIH WFDB data/annotation | `3da9706fc27cc5d627a31b04b74b908ac0b6e061843a87cbd57801e5ea0fed7b` |
| `results/mitdb/104.atr` | 4948 | MIT-BIH WFDB data/annotation | `4f945bef6a4e30b588d7ebdb0763f5294322aa5632b4bb128ee140c7035463c9` |
| `results/mitdb/104.dat` | 1950000 | MIT-BIH WFDB data/annotation | `aeb19bd805996167c6edd8f1b1a659486c0cbad101f06f1426548784d9634354` |
| `results/mitdb/104.hea` | 378 | MIT-BIH WFDB data/annotation | `46b5d09e2cae7caaf48f61fc7d0f20962035cf057cf675097fc7d64c4796748f` |
| `results/mitdb/105.atr` | 5638 | MIT-BIH WFDB data/annotation | `278d534c1ad952a6d41db963b6b2b00be070d8dd5fa66545caa5c18edcc36497` |
| `results/mitdb/105.dat` | 1950000 | MIT-BIH WFDB data/annotation | `fcd0153dfea69c749194bb73de7db124588857cfa2f6aecb72754894c8afde02` |
| `results/mitdb/105.hea` | 258 | MIT-BIH WFDB data/annotation | `9a024eebf8e9baf630308815e689d7e3eab471ab0069db6aded127c396418d48` |
| `results/mitdb/106.atr` | 5476 | MIT-BIH WFDB data/annotation | `585347e6d1bd3ad89cdb52cb80945878a5e51026d81f871a376bb6e13936f7fa` |
| `results/mitdb/106.dat` | 1950000 | MIT-BIH WFDB data/annotation | `c017bc18b95b1f970bca143a044ee5d54168648b541e47aa656f2737a59c1202` |
| `results/mitdb/106.hea` | 158 | MIT-BIH WFDB data/annotation | `ad5ed3ee5603926a07a09f0d0c9e2a7ff10a705dbafcde3d185317730a8839ab` |
| `results/mitdb/107.atr` | 4406 | MIT-BIH WFDB data/annotation | `559453fc0454d2aa2b744224fd301007b8e41585c6b6f239709c8c2cdbe207f4` |
| `results/mitdb/107.dat` | 1950000 | MIT-BIH WFDB data/annotation | `ea2254006bb0503926bfb7f6ca3ccf0b3e1ae6bfa8f5eb702bab5d8e960af728` |
| `results/mitdb/107.hea` | 193 | MIT-BIH WFDB data/annotation | `7c9c70b4ccfd3e503b6d4d20eeb508aa89cfd3a5d2b530bc873aba6a2a503dd4` |
| `results/mitdb/108.atr` | 3742 | MIT-BIH WFDB data/annotation | `ea227008ac3bc4ada87807117909e77ae0643d5a8bc09f32b9096a35f140cbf3` |
| `results/mitdb/108.dat` | 1950000 | MIT-BIH WFDB data/annotation | `229a4ef62679dc5c6ea781bfbaec2bd784151fc9f9517f4cdecd533a9431eb20` |
| `results/mitdb/108.hea` | 311 | MIT-BIH WFDB data/annotation | `1c461d963db1883589a975daa80b4104eee6cabb2f5cfec3e0d63b78e7a6a405` |
| `results/mitdb/109.atr` | 5080 | MIT-BIH WFDB data/annotation | `2eef456587bab17263a126063321c040affa29bc4ccfdf2a2bc7fe50709fa1b8` |
| `results/mitdb/109.dat` | 1950000 | MIT-BIH WFDB data/annotation | `267afd6ced7a1fd5a8bcd21fcc2b83287e37e3ef57f32e8a0fe39a6e021951cd` |
| `results/mitdb/109.hea` | 195 | MIT-BIH WFDB data/annotation | `3b21402a9a096acb6a2199b25681e40b0dd201494cb023ded514cd9726bb7fad` |
| `results/mitdb/111.atr` | 4284 | MIT-BIH WFDB data/annotation | `372ed78bb0bbe7d5d9dc7e676c03d72530729775b5cfa94501485329fc47b8cb` |
| `results/mitdb/111.dat` | 1950000 | MIT-BIH WFDB data/annotation | `8372d98ae59fbfd023502a9d0fc03fb5383083da92584b964484e18593e08485` |
| `results/mitdb/111.hea` | 285 | MIT-BIH WFDB data/annotation | `9e635a715d6cc44a716a7a73c6dd365bd94099f94e63accb2a65c5b25b4b69b5` |
| `results/mitdb/112.atr` | 5118 | MIT-BIH WFDB data/annotation | `69cf7ee24d65a5430162ae9b6452be9d4855c2af47f91976b7f02038ea115002` |
| `results/mitdb/112.dat` | 1950000 | MIT-BIH WFDB data/annotation | `3c7af9758687314255b5d946a58d9509d53ecf2a300230684fa8ecf00b774bdb` |
| `results/mitdb/112.hea` | 198 | MIT-BIH WFDB data/annotation | `a78dc4b0907b3db806395cff76b3a2e9a70e889b9dc3702c7de55aa8b41abd7d` |
| `results/mitdb/113.atr` | 3600 | MIT-BIH WFDB data/annotation | `39527d597a246410d2c16f062600c73271a73a7503d49cafd1d17508af4cd432` |
| `results/mitdb/113.dat` | 1950000 | MIT-BIH WFDB data/annotation | `f5509b71dabaaa11c53f687ee17aab0071bce3d471e2ab77c71a6434f5b7aacf` |
| `results/mitdb/113.hea` | 235 | MIT-BIH WFDB data/annotation | `848ed5243adaf3b53c72b36b1a5a8b5b8cf585a1d78a02e7b25c1d38818d4343` |
| `results/mitdb/114.atr` | 3878 | MIT-BIH WFDB data/annotation | `5280acb082b654d64944137987a00fbc2105b3a3fcdda75d0bf11d4644e40d9f` |
| `results/mitdb/114.dat` | 1950000 | MIT-BIH WFDB data/annotation | `2ae83463edcfec256b79ce93716dcb40f289bacbe9e5e7a66cd00d0a31099d4b` |
| `results/mitdb/114.hea` | 160 | MIT-BIH WFDB data/annotation | `830ba89cebf60945db0c78332e3cc6c64acbe218f6f16594660d4f17423ee889` |
| `results/mitdb/115.atr` | 3946 | MIT-BIH WFDB data/annotation | `4980667b19b7a637729d40762f071b483d69f5431db818cbddc9a8ae64cc4da2` |
| `results/mitdb/115.dat` | 1950000 | MIT-BIH WFDB data/annotation | `34185efe9889608c76ebfbf5a467d3457970c6d11b77a2bf81b94bcf46e07c41` |
| `results/mitdb/115.hea` | 126 | MIT-BIH WFDB data/annotation | `3212847a692bf3bc5e1dde0343d272e7d56735983276031420060d51d0cdd41c` |
| `results/mitdb/116.atr` | 4858 | MIT-BIH WFDB data/annotation | `abd31e2c878073e67319d5a6fa34923fe117cca02e5c80318378405c853ccbef` |
| `results/mitdb/116.dat` | 1950000 | MIT-BIH WFDB data/annotation | `0ab38cfca76d816673213dcd570504bb3ab02ef76d51540fd7a6075bc580ad85` |
| `results/mitdb/116.hea` | 158 | MIT-BIH WFDB data/annotation | `5553273cafa68c106468ddfaa3bd1d6a30daec53cff1642d3592a20110b2edad` |
| `results/mitdb/117.atr` | 3090 | MIT-BIH WFDB data/annotation | `232beaacf4a4eb4b6d2f2242f01ecae3d2842be1dd0afe890037ae050f91ce63` |
| `results/mitdb/117.dat` | 1950000 | MIT-BIH WFDB data/annotation | `2ce7de6f20baf4cfef0369f9a13fa53784d18e2da78c11afa1e4c55e8ad1900e` |
| `results/mitdb/117.hea` | 127 | MIT-BIH WFDB data/annotation | `452e70b5eb3fe3ebc3fc61284f70b788dbf83bb98f951c2b0bb5f0b95156d06a` |
| `results/mitdb/118.atr` | 4622 | MIT-BIH WFDB data/annotation | `77afd5400648d128223d5029a49c442f5abd65ba997f732489cb8eed39d71551` |
| `results/mitdb/118.dat` | 1950000 | MIT-BIH WFDB data/annotation | `66ea4b70b7d64abd0f49089ae1b8f0f577664c46e5b29ac23b291934e063df15` |
| `results/mitdb/118.hea` | 167 | MIT-BIH WFDB data/annotation | `48a3d0c4e7742ca0e39a2cbb3ea5142a4d41261adbd0bb25651e0fcd9c6aef2b` |
| `results/mitdb/119.atr` | 4812 | MIT-BIH WFDB data/annotation | `fd919c8ea20eed4a22fb37984977f30ce48d382eefd2d90934bf947a88e0fd7b` |
| `results/mitdb/119.dat` | 1950000 | MIT-BIH WFDB data/annotation | `02a286a28e69b441568add915f16a2856b22bab893bf8729ee9654411ed3a5aa` |
| `results/mitdb/119.hea` | 158 | MIT-BIH WFDB data/annotation | `658f01382d7280a06e2f4c06363569d477394a6898c69ee788bb383f6092df63` |
| `results/mitdb/121.atr` | 3774 | MIT-BIH WFDB data/annotation | `dbda472aa08eeb6bbbf775b8123e466a7ee57638caecd156407de3141c9fa71c` |
| `results/mitdb/121.dat` | 1950000 | MIT-BIH WFDB data/annotation | `679f374a7b2557fed04a4bc5648301ce9bdd38e5a4f13b2e63ebbcf7cb3fb482` |
| `results/mitdb/121.hea` | 150 | MIT-BIH WFDB data/annotation | `1838d38ff54fed9b831207761d7321682dad0b693f944e4c3a4e6388c46ca6ff` |
| `results/mitdb/122.atr` | 4970 | MIT-BIH WFDB data/annotation | `34966d92a94c72126253c0297407cff78e53be1f72c6814e28676bcda8be90de` |
| `results/mitdb/122.dat` | 1950000 | MIT-BIH WFDB data/annotation | `a700c794bfe23565603d75dc5706915f5436e522802d260e758683d11f5b9e13` |
| `results/mitdb/122.hea` | 221 | MIT-BIH WFDB data/annotation | `b43ca64ed2c6e69d15def2c30f8cb2cbf25cb807f9530b1e7d5cfed2fd496f70` |
| `results/mitdb/123.atr` | 3046 | MIT-BIH WFDB data/annotation | `061bfd2590fd9d6040d32741a19697491dedb65cbbc5b3865a5c50d49f52d709` |
| `results/mitdb/123.dat` | 1950000 | MIT-BIH WFDB data/annotation | `8a685a846b82d72aac4dd86171ad0fe5b19c0bd9ce51ddc1e6f015f73b2d8c3a` |
| `results/mitdb/123.hea` | 183 | MIT-BIH WFDB data/annotation | `aab5351e9da6f4beb22392f1a469df4a9e4b8017fea820432cf7d1846244d094` |
| `results/mitdb/124.atr` | 3360 | MIT-BIH WFDB data/annotation | `f9c8cac368f1f6ba8895ae47f5771c82597374c1e6b030385ca2eb186454677c` |
| `results/mitdb/124.dat` | 1950000 | MIT-BIH WFDB data/annotation | `296988e49d2653eb499ad0b17c46826b4296b7eaf94df3f76f1ba2dcdf6bc48b` |
| `results/mitdb/124.hea` | 219 | MIT-BIH WFDB data/annotation | `2200f917fe7a1fa817b79069a274d3291a6fa66289521af5e7d38edeb3e82bce` |
| `results/mitdb/200.atr` | 8114 | MIT-BIH WFDB data/annotation | `f9624a11696760427d75314a78c31f89ca3c446af855890c8f8b66cddd8b3a3f` |
| `results/mitdb/200.dat` | 1950000 | MIT-BIH WFDB data/annotation | `a9e203b3807b9fcd3647cde03444437cb8eec7f5128a8eb413edafb394272e0f` |
| `results/mitdb/200.hea` | 306 | MIT-BIH WFDB data/annotation | `9e0c2ff5b790cf624deab0ccb8a9f211a9e29a748d8197da3c1ee7c5b596b40c` |
| `results/mitdb/201.atr` | 4322 | MIT-BIH WFDB data/annotation | `df1276b6c6952b177ec6eea2be3edc7a96888454107a3ee56be880b0afaf6f35` |
| `results/mitdb/201.dat` | 1950000 | MIT-BIH WFDB data/annotation | `16610ae46eff900794597503d15af4572a8b1518f18dbed2843b7e68c81f3898` |
| `results/mitdb/201.hea` | 284 | MIT-BIH WFDB data/annotation | `dba34d6252879af28c1d10cde200989a4f6507d98df4395f5cba5f81af5b7e5a` |
| `results/mitdb/202.atr` | 4388 | MIT-BIH WFDB data/annotation | `16d59e44564844c39a648164580a69e71f1657e951e6ad121b64eb1a11db037d` |
| `results/mitdb/202.dat` | 1950000 | MIT-BIH WFDB data/annotation | `0c990e4142de6be898de4b21bd6e6705cd567e5d97cb731a994dc5aee9c7142d` |
| `results/mitdb/202.hea` | 277 | MIT-BIH WFDB data/annotation | `6822216d1bc51f7014637c53e543cdc3863fea4ee3fec3aa99fb0762aece0cba` |
| `results/mitdb/203.atr` | 7094 | MIT-BIH WFDB data/annotation | `2573f948e54a538983a44bf946e3e61cdda60e898b4c65ca8559b57c4877c706` |
| `results/mitdb/203.dat` | 1950000 | MIT-BIH WFDB data/annotation | `5ba7d2237fc838438efd8bb0a3baeffc5de2e70c76e3ab7e401e811ad82f43e0` |
| `results/mitdb/203.hea` | 423 | MIT-BIH WFDB data/annotation | `25b3a3107dd00221b245ef5c23ffdfbd07caaf58e2cf377734f3a947361b26a5` |
| `results/mitdb/205.atr` | 5436 | MIT-BIH WFDB data/annotation | `0ae55e31175b9ba36e09c3b556b15e3e70f254936a2634d5ff341dfb86a74ea9` |
| `results/mitdb/205.dat` | 1950000 | MIT-BIH WFDB data/annotation | `8a513c39e6d9a1d732de0b3909a87aa1fcfd4bd55ab5f59a5b810967d252af36` |
| `results/mitdb/205.hea` | 222 | MIT-BIH WFDB data/annotation | `739f76bbcbbfdad1f260a043a5f7106f6e912894065b15ea75bee42af0380a2a` |
| `results/mitdb/207.atr` | 4958 | MIT-BIH WFDB data/annotation | `cceb64d68033a277d2d5669458d49258295102dd3e20614a0ca7d63b67009404` |
| `results/mitdb/207.dat` | 1950000 | MIT-BIH WFDB data/annotation | `139f99250366fbf347cba4d8ea1fbe788f98cb1b93b70f50ccba70122d908605` |
| `results/mitdb/207.hea` | 546 | MIT-BIH WFDB data/annotation | `7645d488d4c304760aae0a709193ffa13692c317b551bcbdcbb37011032178d8` |
| `results/mitdb/208.atr` | 8578 | MIT-BIH WFDB data/annotation | `33878519ee34d00cb233fdb15b11c2b8da0b0aea2e643d31b07105cc3392edf4` |
| `results/mitdb/208.dat` | 1950000 | MIT-BIH WFDB data/annotation | `22873623623c44ee413a5b60cacb15d4af2723734836fb3ae45d9ff96bdb36e7` |
| `results/mitdb/208.hea` | 308 | MIT-BIH WFDB data/annotation | `5a27efda1bea972440a0b5f40a1c322898693e9602d5ce4722fe750b7dea46e9` |
| `results/mitdb/209.atr` | 6288 | MIT-BIH WFDB data/annotation | `c0697b176e7caf49e34f93e28eaeeff7167e2de380a8698f6cb3fb3a5fcb1c84` |
| `results/mitdb/209.dat` | 1950000 | MIT-BIH WFDB data/annotation | `e2516dd126ea85529d9a0e8442fef5d880a492d41f0692a76e648c8437f15c5f` |
| `results/mitdb/209.hea` | 157 | MIT-BIH WFDB data/annotation | `9ac1abedc17f898bf6a1db9c64c38ccf4f065ebc3989ee7740f4c0796655e590` |
| `results/mitdb/210.atr` | 5824 | MIT-BIH WFDB data/annotation | `5a46bef3d185c903b1ce099a988898fc829cea7e6c5bc1984532efd694fa8db6` |
| `results/mitdb/210.dat` | 1950000 | MIT-BIH WFDB data/annotation | `76e971fba03c0ae2ab261c7446240987ecde82f45dad1d4f36e290b950eb464b` |
| `results/mitdb/210.hea` | 158 | MIT-BIH WFDB data/annotation | `9d6a422834435fb8d267256f4fd01bfd8158b511f662c4b5dd8c60059944a322` |
| `results/mitdb/212.atr` | 5550 | MIT-BIH WFDB data/annotation | `dc4d0bec7e3ced82f67ef9d5264bcdb3d167c2214125eb04891040fbbb340816` |
| `results/mitdb/212.dat` | 1950000 | MIT-BIH WFDB data/annotation | `6b63d3e20c40073be852e8121196d5de2ede502168fcb6c10ca5af841418cc5c` |
| `results/mitdb/212.hea` | 248 | MIT-BIH WFDB data/annotation | `f67d7c99a8ba055941e8aa98fd1f31dfd7dd1d957c85f40b488237eecba61aad` |
| `results/mitdb/213.atr` | 6852 | MIT-BIH WFDB data/annotation | `44349e22176aad3f688ca9a62f283a993927a0b9ef6d97f473cde3b16cfabc9a` |
| `results/mitdb/213.dat` | 1950000 | MIT-BIH WFDB data/annotation | `0cb13cd9299b0ab68f8f3064b2ab72f853d155abb277b68daef0a26cbc173b0d` |
| `results/mitdb/213.hea` | 327 | MIT-BIH WFDB data/annotation | `6b64b05c7a28630acf3ffb5e541d635e3fc854ea3602b1b355762f94c6058f17` |
| `results/mitdb/214.atr` | 4790 | MIT-BIH WFDB data/annotation | `09c588b0623f6dc6138928ce2006f294c4ef3da143eec2cdba6eddb9b1597214` |
| `results/mitdb/214.dat` | 1950000 | MIT-BIH WFDB data/annotation | `a886b1ceb56e950858dedf5b56d3fdb5f26aef3667f8effcb880c1c000ee0cd3` |
| `results/mitdb/214.hea` | 270 | MIT-BIH WFDB data/annotation | `beb6b379e93cbe94eb9f1461cccaa0131940e14322df4abbc07665b722ed9206` |
| `results/mitdb/215.atr` | 6878 | MIT-BIH WFDB data/annotation | `94968307b89eb0be766cbb88617f08571ed41b3b25b294d19a72b7ffd2525aee` |
| `results/mitdb/215.dat` | 1950000 | MIT-BIH WFDB data/annotation | `9e28799b5733a2d60470653e6f6ca26610143303233a1b7559cff54e2ad26ef9` |
| `results/mitdb/215.hea` | 256 | MIT-BIH WFDB data/annotation | `a326c25698a0773a6be986f2cae499981224d3c61d8fdcddfbb3beaf3ff0bcbd` |
| `results/mitdb/217.atr` | 5262 | MIT-BIH WFDB data/annotation | `196ad0989b837b5d51784e13bef1242549ca7266f0da21743f208302e3e3f5e1` |
| `results/mitdb/217.dat` | 1950000 | MIT-BIH WFDB data/annotation | `de29975a2a476eaa1a3f07b90134cec70830b2d7932b213354ba175914a6040a` |
| `results/mitdb/217.hea` | 178 | MIT-BIH WFDB data/annotation | `5320a16d3a7d1dea32cdf905b7f5140772a01bf6b11065e0ab937cc6775c5ce6` |
| `results/mitdb/219.atr` | 4798 | MIT-BIH WFDB data/annotation | `271928544ce78772669c50b26cf4de163a53f5b90187b5d8080928def79f0e6a` |
| `results/mitdb/219.dat` | 1950000 | MIT-BIH WFDB data/annotation | `7fc27b77e313aaa40da9f39783f8fb29a6a910be148e67463e5093793644751a` |
| `results/mitdb/219.hea` | 276 | MIT-BIH WFDB data/annotation | `d14f0ffbf19f9842dbaa15f82bcedbc7b9b96f00f422b7e4da31112a1d1b2b67` |
| `results/mitdb/220.atr` | 4262 | MIT-BIH WFDB data/annotation | `9cf9f330823e2a9850d51a7fce1f8ecfe5bb84e80e89412bd9f4de6a67f9ccc7` |
| `results/mitdb/220.dat` | 1950000 | MIT-BIH WFDB data/annotation | `c01ca1a76fb3a89e96134a08f2e5aa6e760eda59ea9086ba8aaa13ba0cca9cff` |
| `results/mitdb/220.hea` | 132 | MIT-BIH WFDB data/annotation | `af40adc6deb9c0e942432e69a7153e41ac252eb6c15a24a715ef258469cda2e5` |
| `results/mitdb/221.atr` | 5874 | MIT-BIH WFDB data/annotation | `92549b966cc17bcf2edf00a97237d1e350a761d9ffe20065b5ed303848aad7ae` |
| `results/mitdb/221.dat` | 1950000 | MIT-BIH WFDB data/annotation | `274d17c6a35f7adcbb10a59ea34350e5ad299c2d69e4fa579d8863213e1fe9ae` |
| `results/mitdb/221.hea` | 228 | MIT-BIH WFDB data/annotation | `09afd18974ee6e0cfa8b5b6ce80112c4df0cfbb8c8d01d3d085ecdf7790ada7b` |
| `results/mitdb/222.atr` | 6230 | MIT-BIH WFDB data/annotation | `a0206b778a9f073d451023e802e52059754648e83fe1dc669395b4884ccad356` |
| `results/mitdb/222.dat` | 1950000 | MIT-BIH WFDB data/annotation | `0510bf39ed2925a83debd249492bdfe77c575148376aaabeba4c7b7a2451ac5d` |
| `results/mitdb/222.hea` | 333 | MIT-BIH WFDB data/annotation | `a395bf5e58492e6724616532aae1cca7c50626fd66fe61f843a92adadc27293d` |
| `results/mitdb/223.atr` | 5468 | MIT-BIH WFDB data/annotation | `beaa5cb3eb5fb64a5f9e1dfcac9297f9564b379c9c354cbf8f9f927d9b81ccf1` |
| `results/mitdb/223.dat` | 1950000 | MIT-BIH WFDB data/annotation | `1d42aa9e13f7349287ae5f0f00b40078c01aadd39e2cad1001ab42d3b7484594` |
| `results/mitdb/223.hea` | 258 | MIT-BIH WFDB data/annotation | `bbd414e57b6eafa53204fb7eeb1212c9d47f598f36e8838b103bfb68a316ca85` |
| `results/mitdb/228.atr` | 5198 | MIT-BIH WFDB data/annotation | `7e08ee90f668800fb3e1a66193fb8f8b4b91c3ec2f52f384756bab826e019d1d` |
| `results/mitdb/228.dat` | 1950000 | MIT-BIH WFDB data/annotation | `b2421f15447772bb7a38f5073e4e07d5d9be50d875441a4a0821c9e22b78fdac` |
| `results/mitdb/228.hea` | 302 | MIT-BIH WFDB data/annotation | `b4a2796ca39816b32d5b24c2c1d0a2aa355d159aafd7ecbfb1cdf11a695a14d8` |
| `results/mitdb/230.atr` | 6386 | MIT-BIH WFDB data/annotation | `403d0d4842ba7135b1e14b9e5c50cba214753a0e17bf979090b2c6a8018f7eae` |
| `results/mitdb/230.dat` | 1950000 | MIT-BIH WFDB data/annotation | `fba09ea54d9caa342c54c46e0df247335345b02444f7d15162906adfef55d057` |
| `results/mitdb/230.hea` | 133 | MIT-BIH WFDB data/annotation | `5738bb479316dcc324b27807bbbffd8878da6156707c6cfbc08dcd0fa4771671` |
| `results/mitdb/231.atr` | 7516 | MIT-BIH WFDB data/annotation | `b82cc005618eab81b3a246ca5c5a61903f3c0f40c191d7fb953adf054ea691de` |
| `results/mitdb/231.dat` | 1950000 | MIT-BIH WFDB data/annotation | `7d4f56d6c4b1d50322154381c1d8761457c2f2696973e915a3c19d8a5b789595` |
| `results/mitdb/231.hea` | 333 | MIT-BIH WFDB data/annotation | `733195a792e806a897531475eabcdec71e96f7b6cd7bcf57f9f1b2538cb13503` |
| `results/mitdb/232.atr` | 3914 | MIT-BIH WFDB data/annotation | `be0e236cb883c669128da01beae0ec0efc39fdfaa5c1e878a3cef5443db8c934` |
| `results/mitdb/232.dat` | 1950000 | MIT-BIH WFDB data/annotation | `3d74a21f848766c9607bc5eead7e4b2655f09586adb5561f02fd8fcb0e9da6d0` |
| `results/mitdb/232.hea` | 389 | MIT-BIH WFDB data/annotation | `5ff480450238ffc957abde76687b7e3d728c4b7b19063c8a0b1d08469a2228ea` |
| `results/mitdb/233.atr` | 6770 | MIT-BIH WFDB data/annotation | `c9ce42ecefa9a92e2ec69f9d237d8a53c67c60b31a25d579356ea8910bcc368a` |
| `results/mitdb/233.dat` | 1950000 | MIT-BIH WFDB data/annotation | `6ccd6b1fa0cafc052369e7ec0a99cc6c3b56b3c12afff26db800c55a826211b4` |
| `results/mitdb/233.hea` | 162 | MIT-BIH WFDB data/annotation | `d5ef73b3b4a98bec6e3d853db08ff18b0a1a9173c95327bcb15b2cfecee99b85` |
| `results/mitdb/234.atr` | 5560 | MIT-BIH WFDB data/annotation | `e829a3e4131fbe9420c1740bfb5000b3f4f64a2b40aa5f739ed6420f75be6785` |
| `results/mitdb/234.dat` | 1950000 | MIT-BIH WFDB data/annotation | `89f826307bfc91a554021da87c17b88b843ed3d26aa9ea237fc6f7c8af00da6d` |
| `results/mitdb/234.hea` | 156 | MIT-BIH WFDB data/annotation | `91c94c803d05f637cc29a3f353d88e1c25b20a25957c8cc25f7cbbf9fe2627c9` |
| `results/multi_seed_test.csv` | 489 | Metrics/history/audit CSV | `1926e305d04d65505cfab2a1aaae8411c7acf8218ced45591ad5e8db4ea99ded` |
| `results/neurocardio_final.keras` | 1794687 | Keras model | `0e7083d56c31a4e43f6cd332b375b8480446485b60b80f52c3c7c5380508becc` |
| `results/patient_class_distribution.csv` | 1162 | Metrics/history/audit CSV | `0cb5bcbc864daba3e7b6c7cd89b8f01187127ee7e83e143e09d84e48e8dbc1cf` |
| `results/per_record_test.csv` | 1178 | Metrics/history/audit CSV | `357cd26bd1a47ea0ad518d90d14d203e97049f5aab2f2b146fe9649da349b5f0` |
| `results/pr_curves_cv.png` | 61982 | Plot, XAI, or notebook output image | `91a534fd3f2a2d563a9feee163a1dcd9fe23ca863ba9e3b3ffad55fa79ef219c` |
| `results/pr_curves_test.png` | 65684 | Plot, XAI, or notebook output image | `aceebb96c61bd098f11c50d9ea4a80edf82ebee992280c076209dd7b716942e9` |
| `results/preprocessing_config.json` | 4259 | Configuration or metrics JSON | `0177a64052b5d5fb260b271b8069bab65c1265da791d57c9f9edb71884aedae4` |
| `results/selective_operating_points.csv` | 249 | Metrics/history/audit CSV | `a8967be40a8b6e465defc07f8a69dbfff2f718dc79caf657ba97a711afa954d9` |
| `results/selective_test_balanced.csv` | 322 | Metrics/history/audit CSV | `07b05a02ae249faab376306d0bb4dadd4efbe6d1a49f8d7521ed4a126f8260c6` |
| `results/selective_test_conservative.csv` | 277 | Metrics/history/audit CSV | `fc8f00b90abb682f51be9b3aa5e5b7ae26d324d35c9a7eb19d0d87b5534c7550` |
| `results/selective_test_permissive.csv` | 341 | Metrics/history/audit CSV | `50d4807dd8e27de9837b5cfbef83ce154a6b130f5f321384959b42b812820e0d` |
| `results/selective_thresholds.json` | 655 | Configuration or metrics JSON | `8e83122cddf10ab36c869ebae789db2b7180992c58347eff80bb11e001801b07` |
| `results/split_selection.json` | 1050 | Configuration or metrics JSON | `ca78334afe445e5d3e251bbad665073552394dc8efd476a36a90ebff22ff0c33` |
| `results/test_examples_for_xai.npz` | 128366561 | NumPy records/state/examples | `7f28287060e089829186c7d7dd77c73154a5e6c708a511619fd102251b037f54` |
| `results/test_metrics.json` | 22010 | Configuration or metrics JSON | `1b0363b1e07553b44110f08b25b5e22096060e464622f27369eb2cea212f8033` |
| `results/test_raw_primary.json` | 667 | Configuration or metrics JSON | `fa514e20544a9557ca84a3d42376d4b63048c7b21bc7427be5f62c24e8c9540d` |
| `results/test_records.npy` | 380 | NumPy records/state/examples | `b47a5febb01ab10ad986fdbabdcb3d8583abba5bfc33647cec9110e2842d0505` |
| `results/test_tuned_ablation.json` | 785 | Configuration or metrics JSON | `901871d7fa10ee666bec86bbc7cb15779377ee5c9079815d01048c4ce2c1d374` |
| `results/train_records.npy` | 320 | NumPy records/state/examples | `b1a64c9a36f5f46e32c6ae97bdc01c51a5a4d273f7023eed38c8947cf77af58e` |
| `results/training_curves.png` | 67866 | Plot, XAI, or notebook output image | `1c70bf1c92ec4b339e87f5d0092aa3e6cc635fa406fc6922e8c7ca42dc537a15` |
| `results/training_history.csv` | 31261 | Metrics/history/audit CSV | `6c33e3a87542a8f7cfd24a015083fc5916e0b05b8f0be61daf023d15c53543b1` |
| `results/validation_records.npy` | 212 | NumPy records/state/examples | `7b0128ea748530a52588cb57166b7d0f125762b3717dba938942d864ffd40198` |
| `results/window_length_audit.csv` | 893 | Metrics/history/audit CSV | `fb2bd980f3686bc329692202aa47a3552743559a4248efcfd51c940ebb101603` |
| `results/xai_examples.png` | 532722 | Plot, XAI, or notebook output image | `447189cd8acfd7bbd8931921add15e91b19a264be5e52191c28558fd686b670e` |
