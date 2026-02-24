from kokoro_onnx import Kokoro
import soundfile as sf
import time

kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

prompt = """
The reports of my death are greatly exaggerated, but the reports of AI taking all our jobs... are somehow even more so.
"""

# List available voices: print(kokoro.get_voices())

voice = 'af_bella'

# Token approximation (words * 1.3 is a rough heuristic)
words = len(prompt.split())
approx_tokens = int(words * 1.3)

print(f"Prompt      : {prompt.strip()}")
print(f"Word count  : {words}")
print(f"~Tokens     : {approx_tokens}")
print(f"Voice       : {voice}")
print(f"Generating...")

start = time.perf_counter()
audio, sr = kokoro.create(prompt, voice=voice, speed=1.0, lang="en-us")
end = time.perf_counter()

duration_generated = len(audio) / sr  # audio length in seconds
inference_time = end - start
rtf = inference_time / duration_generated  # real-time factor

print(f"\n--- Timing ---")
print(f"Inference time : {inference_time:.3f}s")

# Save the audio
sf.write('output.wav', audio, sr)
