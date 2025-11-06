# README.md
# EEG XAI Pipeline (MVP)

Pipeline complet:
1) preprocess → 2) segment_rem → 3) extract_features → 4) train → 5) infer → 6) xai_attributions → 7) xai_attention_maps → 8) group_stats → 9) build_reports

Usage rapide:
- make all
- make train
- make xai_attr

Chemins:
- Données brutes lues depuis $EEG_RAW_ROOT (ou à configurer dans configs/*).
