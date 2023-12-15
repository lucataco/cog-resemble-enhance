# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
import torchaudio
from typing import List
from resemble.resemble_enhance.enhancer.inference import denoise, enhance

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        

    def predict(
        self,
        input_audio: Path = Input(description="Input audio file"),
        solver: str = Input(
            description="Solver to use",
            default="Midpoint",
            choices=["Midpoint", "RK4", "Euler"]
        ),
        number_function_evaluations: int = Input(
            description="CFM Number of function evaluations to use",
            default=64, ge=1, le=128
        ),
        prior_temperature: float = Input(
            description="CFM Prior temperature to use",
            default=0.5, ge=0, le=1.0
        ),
        denoise_flag: bool = Input(
            description="Denoise the audio",
            default=False
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        solver = solver.lower()
        nfe = int(number_function_evaluations)
        lambd = 0.9 if denoise_flag else 0.1

        dwav, sr = torchaudio.load(str(input_audio))
        dwav = dwav.mean(dim=0)

        wav1, new_sr1 = denoise(dwav, sr, device)
        wav2, new_sr2 = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=prior_temperature)

        wav1 = wav1.unsqueeze(0)
        wav2 = wav2.unsqueeze(0)

        outputs = []
        output_path1 = "/tmp/output-denoised.wav"
        output_path2 = "/tmp/output-enhanced.wav"
        torchaudio.save(output_path1, wav1, new_sr1)
        torchaudio.save(output_path2, wav2, new_sr2)
        outputs.append(Path(output_path1))
        outputs.append(Path(output_path2))

        return outputs
