# AttentionTrack Platform

## repository structure

The repository should use the following layout for the current working setup:

```text
AttentionTrack-Platform/
├─ .venv/                                  # local virtual environment only
├─ apps/
│  ├─ __init__.py
│  └─ streamlit_app_instructor_dashboard_high_contrast.py
├─ attentiontrack/
│  ├─ __init__.py
│  ├─ dataset.py
│  ├─ feature_extractor.py
│  ├─ model.py
│  ├─ model_ns.py
│  └─ dual_agents_langgraph_llm.py
├─ scripts/
│  ├─ __init__.py
│  ├─ make_stratified_split_ns.py
│  ├─ baseline_tabular.py
│  ├─ train_lstm_3.py
│  ├─ train_xgboost.py
│  ├─ train_tcn.py
│  ├─ train_gru.py
│  ├─ train_transformer_encoder.py
│  ├─ live_attentiontrack.py
│  ├─ live_attentiontrack_record_blur_fast.py
│  └─ live_attentiontrack_langgraph_llm_record_blur_bilstm.py
├─ data/
│  ├─ raw/
│  │  └─ daisee/
│  │     ├─ train/
│  │     ├─ val/
│  │     └─ test/
│  └─ processed/
│     └─ daisee_split/
│        ├─ train/
│        ├─ val/
│        ├─ test/
│        └─ split_manifest.csv
├─ artifacts/
│  ├─ baselines/
│  │  ├─ daisee/
│  │  │  └─ baseline_metrics.json
│  │  └─ xgboost/
│  │     └─ daisee_resplit_v1/
│  │        ├─ xgboost_model.pkl
│  │        └─ metrics.json
│  └─ models/
│     ├─ lstm/
│     ├─ bilstm/
│     ├─ tcn/
│     ├─ gru/
│     └─ transformer_encoder/
├─ recordings/
│  └─ daisee/
├─ logs/
│  └─ daisee/
├─ attention_events.jsonl
├─ requirements.txt
└─ README.md
```

## Rules

- Never place project code inside `.venv/Scripts`.
- Put runnable modules inside `scripts/` and run them from the repo root with `python -m scripts.<module>`.
- Put shared code inside `attentiontrack/`.
- Use `data/raw/<dataset>/` as the input NPZ root.
- Use `data/processed/<dataset>_split/` as the resplit output.
- Use `artifacts/` for model outputs and metrics.
- Use `logs/` for JSONL logs. A root-level `attention_events.jsonl` file is supported and currently works.

## Split command

```powershell
python -m scripts.make_stratified_split_ns `
  --features_dir data\raw\daisee `
  --manifest_csv data\processed\daisee_split\split_manifest.csv `
  --copy_to_dir data\processed\daisee_split `
  --seed 42
```

## Baseline and training

```powershell
python -m scripts.baseline_tabular `
  --features_dir data\processed\daisee_split `
  --save_dir artifacts\baselines\daisee `
  --decision_metric bal_acc `
  --thr_min 0.20 `
  --thr_max 0.90 `
  --thr_steps 15
```

```powershell
python -m scripts.train_lstm_3 `
  --features_dir data\processed\daisee_split `
  --save_dir artifacts\models\bilstm\daisee_resplit_v1 `
  --epochs 30 `
  --batch_size 64 `
  --lr 1e-3 `
  --hidden_size 128 `
  --num_layers 2 `
  --dropout 0.3 `
  --patience 7 `
  --pos_weight_scale 0.3 `
  --decision_metric bal_acc `
  --thr_min 0.20 `
  --thr_max 0.90 `
  --thr_steps 15 `
  --bidirectional
```

## Live recording and dashboard

```powershell
python -m scripts.live_attentiontrack_record_blur_fast `
  --model_path artifacts\models\bilstm\daisee_resplit_v1\best.pt `
  --norm_path artifacts\models\bilstm\daisee_resplit_v1\norm.json `
  --record_path recordings\daisee\session01.mp4 `
  --blur_mode background `
  --events_path attention_events.jsonl `
  --student_id "Student B"
```

```powershell
python -m streamlit run .\apps\streamlit_app_instructor_dashboard_high_contrast.py
```

Dashboard source path:

```text
attention_events.jsonl
```

## Compatibility note

There are two model families in the repository:

- `attentiontrack.model` -> used by `train_lstm_3.py` and `live_attentiontrack.py`
- `attentiontrack.model_ns` -> used by `live_attentiontrack_record_blur_fast.py` and `live_attentiontrack_langgraph_llm_record_blur_bilstm.py`

Do not assume checkpoints are interchangeable between these two families unless loading has been verified successfully.
