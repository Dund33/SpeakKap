from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torchaudio


class WeSpeakerService:

    def __init__(
        self,
        model_path: str = "models/voxceleb_resnet293_LM.onnx"
    ):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

    def get_embedding(
        self,
        audio_path: str
    ) -> np.ndarray:

        waveform, sample_rate = torchaudio.load(
            audio_path
        )

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        # Resample to 16kHz
        if sample_rate != 16000:

            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )

            waveform = resampler(waveform)

        audio = waveform.squeeze(0).numpy()

        # Shape: [1, T]
        audio = np.expand_dims(
            audio,
            axis=0
        ).astype(np.float32)

        embedding = self.session.run(
            None,
            {
                self.input_name: audio
            }
        )[0]

        return np.asarray(
            embedding[0],
            dtype=np.float32
        )