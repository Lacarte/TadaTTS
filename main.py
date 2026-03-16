import torch
import torchaudio
import soundfile as sf
import time

from tada.modules.tada import TadaForCausalLM
from tada.modules.encoder import Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"Device: {device}")
print("Loading encoder...")
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
print("Loading TADA-1B model...")
model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b", torch_dtype=dtype).to(device)

# Load a reference audio for voice cloning
ref_audio_path = "reference.wav"
print(f"Loading reference audio: {ref_audio_path}")
audio, sample_rate = torchaudio.load(ref_audio_path)
audio = audio.to(device)

ref_transcript = "This is a sample reference transcript."

prompt = encoder(audio, text=[ref_transcript], sample_rate=sample_rate)

text_to_speak = """
The reports of my death are greatly exaggerated, but the reports of AI taking all our jobs... are somehow even more so.
"""

print(f"Text: {text_to_speak.strip()}")
print("Generating...")

start = time.perf_counter()
output = model.generate(prompt=prompt, text=text_to_speak.strip())
end = time.perf_counter()

inference_time = end - start
generated_audio = output.audio[0].cpu()
duration = generated_audio.shape[-1] / 24000
rtf = inference_time / duration

print(f"\n--- Timing ---")
print(f"Inference time : {inference_time:.3f}s")
print(f"Audio duration : {duration:.2f}s")
print(f"RTF            : {rtf:.3f}")

torchaudio.save("output.wav", generated_audio.unsqueeze(0), 24000)
print("Saved to output.wav")
