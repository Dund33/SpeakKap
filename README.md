# SpeakKap
Yet another speaker recognition system

## dataset and experiments

The experiment harness lives in `experiments/`. It uses the existing Flask API:

1. Put a speaker dataset on disk as one directory per speaker:

   ```text
   data/dataset/
     speaker_001/*.wav
     speaker_002/*.wav
     ...
   ```

   VoxCeleb, CNCeleb, VoxBlink subsets, or your own scraped recordings can be used after conversion to a supported audio format. The runner requires at least 100 speakers and at least 5 files per speaker by default: 4 enrollment files and at least 1 unseen test file.

2. Start Redis and the API:

   ```powershell
   docker compose up redis
   python -m src.api
   ```

3. Run the experiments:

   ```powershell
   python -m experiments.run_experiments `
     --dataset-root data/dataset `
     --background-root data/background_noise `
     --rir-root data/room_impulse_responses `
     --output-dir experiments/results
   ```

The runner writes `manifest.csv`, one CSV per experiment, and one metrics JSON per experiment. It covers the PDF-required baseline, amplitude scaling, naive/interpolated downsampling, Gaussian noise, background noise, codec compression, and reverberation experiments.

Codec experiments require `ffmpeg` in `PATH`. Use `--skip-codec` when ffmpeg is not installed.
