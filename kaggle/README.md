# Kaggle Phase 2 runner

Upload `lma_phase2_kaggle_upload.zip` as a private Kaggle dataset and attach that dataset to a GPU
notebook. Open the included `phase2_kaggle.ipynb`; it packs the raw splits in Kaggle working
storage before starting training.

Training Nepali:

```bash
python /path/to/bundle/kaggle/run_phase2_kaggle.py --language nepali --mode train
```

Use `--language all` to run every configured model. Checkpoints are written to
`/kaggle/working/phase2_runs`. Before a session ends, save the notebook output or create a private
Kaggle dataset from that directory. Attach the previous output on the next session and pass its
path with `--resume-root`; the runner copies it to writable storage and resumes `latest`.

After training completes, install `sentencepiece` and `sacrebleu` if the Kaggle image does not
already provide them, then run:

```bash
python /path/to/bundle/kaggle/run_phase2_kaggle.py --language nepali --mode evaluate
```

Evaluation JSON, samples, attention summaries, heatmaps, and the combined training curve are
written under `/kaggle/working/phase2_report`.

## Nepali Phase 3

Attach the repository bundle, the generated `reasoning_v1` data, and the Phase 2 artifact folder.
One command runs the 20k main finetune, both held-out evaluation protocols, and the balanced
1k/3k/5k/8k/12k sample-size ablations:

```bash
python kaggle/run_phase3_nepali.py \
  --artifact-root /kaggle/input/lma-artifacts/LMA_Individual_Project \
  --dataset-root /kaggle/input/lma-phase3-reasoning/reasoning_v1 \
  --work-root /kaggle/working/phase3_nepali \
  --mode all --device cuda
```

Use `--mode train --resume` after attaching a saved prior output. Modes `evaluate` and `ablate`
can also run independently once `best_checkpoint.pt` is restored under the selected work root.
