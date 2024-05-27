# This approach did not work: the audio model is broken, onnx does not run naively on MacOS and the ONNX-to-CoreML conversion is deprecated.

# %%
import start  # noqa isort:skip

# %%
import onnx
import onnxruntime
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
onnx_audio_encoder = torch.onnx.dynamo_export(model.clap.audio_encoder, audio_inputs)

# %%
onnx_audio_encoder.save("audio_encoder.onnx")

# %%
onnx_text_encoder = torch.onnx.dynamo_export(model.clap.caption_encoder, text_inputs)

# %%
onnx_text_encoder.save("text_encoder.onnx")

# %%
# Check ONNX models
onnx_audio_model = onnx.load("audio_encoder.onnx")
onnx_text_model = onnx.load("text_encoder.onnx")
onnx.checker.check_model(onnx_audio_model)
onnx.checker.check_model(onnx_text_model)

# %%
# Execute ONNX text model with ONNX Runtime
onnx_text_input = onnx_text_encoder.adapt_torch_inputs_to_onnx(text_inputs)
print(f"Input length: {len(onnx_text_input)}")
print(f"Sample input: {onnx_text_input}")

ort_session = onnxruntime.InferenceSession(
    "./text_encoder.onnx", providers=["CPUExecutionProvider"]
)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


onnxruntime_input = {
    k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_text_input)
}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

# %%
# Compare the PyTorch results with the ones from the ONNX Runtime
torch_outputs = model.clap.caption_encoder(text_inputs)
torch_outputs = onnx_text_encoder.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")


# %%
# Execute ONNX audio model with ONNX Runtime
onnx_audio_input = onnx_audio_encoder.adapt_torch_inputs_to_onnx(audio_inputs)
print(f"Input length: {len(onnx_audio_input)}")
print(f"Sample input: {onnx_audio_input}")

ort_session = onnxruntime.InferenceSession(
    "./audio_encoder.onnx", providers=["CPUExecutionProvider"]
)

onnxruntime_input = {
    k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_audio_input)
}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

# %%
# Compare the PyTorch results with the ones from the ONNX Runtime
torch_outputs = model.clap.audio_encoder(audio_inputs)
torch_outputs = onnx_audio_encoder.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")

# %%
