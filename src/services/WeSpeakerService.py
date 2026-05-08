from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torchaudio
import torchaudio.compliance.kaldi as kaldi


class WeSpeakerService:
    def __init__(
        self,
        model_path: str = "models/voxceleb_resnet293_LM.onnx",
        device: str = "cpu",
    ):
        providers = (
            ["CUDAExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, audio_path: str) -> np.ndarray:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000,
            )
            waveform = resampler(waveform)

        # Extract 80-dimensional FBANK features
        feats = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=16000,
            use_energy=False,
        )

        # Mean normalization
        feats = feats - feats.mean(dim=0, keepdim=True)

        # Convert to NumPy and add batch dimension
        feats = feats.numpy().astype(np.float32)
        feats = np.expand_dims(feats, axis=0)  # [1, frames, 80]

        # Run inference
        embedding = self.session.run(
            None,
            {self.input_name: feats},
        )[0]

        # Remove batch dimension
        return np.asarray(embedding[0], dtype=np.float32)