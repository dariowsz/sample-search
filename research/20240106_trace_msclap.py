# %%
import research.start as start  # noqa isort:skip

# %%
import coremltools as ct
import torch
from msclap import CLAP

# %%
model = CLAP(
    "checkpoints/CLAP_weights_2023.pth",
    version="2023",
    use_cuda=False,
)

test_text = ["Neo-soul guitar"]
text_inputs = model.preprocess_text(test_text)
test_audio = [
    "/Users/dario.wisznewer/Test_Samples/Cymatics - GG Chillwave Chord Loop 1 - 100 BPM A Maj.wav"
]
audio_inputs = model.preprocess_audio(test_audio, resample=True)

with torch.no_grad():
    audio_inputs = audio_inputs.reshape(audio_inputs.shape[0], audio_inputs.shape[2])  # type: ignore

# %%
traced_model = torch.jit.trace_module(
    model.clap, {"forward": (audio_inputs, text_inputs)}
)
# TODO: Not working yet... Text input is a dict not a Tensor
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(shape=audio_inputs.shape),
        ct.TensorType(shape=text_inputs.shape),  # type: ignore
    ],
)  # convert to coreml

# %%
torch.jit.save(traced_model, "msclap.pt")

# %%
