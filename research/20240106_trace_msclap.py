# %%
import research.start as start  # noqa isort:skip

# %%
import coremltools as ct
import numpy as np
import torch
from msclap import CLAP

# %%
# Initialize model, text and audio inputs
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
# Trace text encoder
traced_text_model = torch.jit.trace(
    model.clap.caption_encoder,
    (text_inputs["input_ids"], text_inputs["attention_mask"]),  # type: ignore
)

# %%
# Convert text encoder to CoreML
# https://github.com/mazzzystar/Queryable/blob/main/PyTorch2CoreML-HuggingFace.ipynb
# FIXME: For some reason there is a validation error if max_seq_length is set to 77
max_seq_length = 77
text_encoder_model = ct.convert(
    traced_text_model,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS12,
    inputs=[
        ct.TensorType(name="input_ids", shape=[1, max_seq_length], dtype=np.int64),
        ct.TensorType(name="attention_mask", shape=[1, max_seq_length], dtype=np.int64),
    ],
    outputs=[
        ct.TensorType(name="text_embedding", dtype=np.float32),
    ],
)
text_encoder_model.save("CLAPTextEncoder_float32.mlpackage")  # type: ignore

# %%
# Validate export presicion
text_model = ct.models.MLModel("coreml_models/CLAPTextEncoder_float32.mlpackage")

test_text_inputs = model.preprocess_text(["Neo-soul guitar"])
input_ids = test_text_inputs["input_ids"].to(torch.int32)  # type: ignore
attention_mask = test_text_inputs["attention_mask"].to(torch.int32)  # type: ignore
# FIXME: Check if this is necessary
# input_ids = input_ids[:, :max_seq_length]
# attention_mask = attention_mask[:, :max_seq_length]

predictions = text_model.predict(
    {"input_ids": input_ids, "attention_mask": attention_mask}
)
model.clap.caption_encoder = model.clap.caption_encoder.eval()
out = model.clap.caption_encoder(input_ids, attention_mask)  # type: ignore

# %%
# NOTE:
print('PyTorch TextEncoder ckpt out for "Neo-soul guitar":\n>>>', out[0, :10])
print(
    '\nCoreML TextEncoder ckpt out for "Neo-soul guitar":\n>>>',
    predictions["text_embedding"][0, :10],
)


# %%
# Trace audio encoder
traced_audio_model = torch.jit.trace(model.clap.audio_encoder, audio_inputs)

# %%
# Convert audio encoder to CoreML
audio_encoder_model = ct.convert(
    traced_audio_model,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS12,
    inputs=[
        ct.TensorType(name="audio", shape=[1, 308700], dtype=np.float32),
    ],
    outputs=[
        ct.TensorType(name="projected_vec", dtype=np.float32),
        ct.TensorType(name="audio_classification_output", dtype=np.float32),
    ],
)
audio_encoder_model.save("CLAPAudioEncoder_float32.mlpackage")  # type: ignore

# %%
# Validate export presicion
audio_model = ct.models.MLModel("coreml_models/CLAPAudioEncoder_float32.mlpackage")

predictions = audio_model.predict({"audio": audio_inputs})
projected_vec, audio_classification_output = model.clap.audio_encoder(audio_inputs)  # type: ignore

# %%
print(
    'PyTorch AudioEncoder ckpt out for "Neo-soul guitar":\n>>>',
    projected_vec[0, :10],
)
print(
    '\nCoreML AudioEncoder ckpt out for "Neo-soul guitar":\n>>>',
    predictions["projected_vec"][0, :10],
)

# %%
audio_predictions = audio_model.predict({"audio": audio_inputs})

# %%
test_text_inputs = model.preprocess_text(["Guitar"])
input_ids = test_text_inputs["input_ids"].to(torch.int32)  # type: ignore
attention_mask = test_text_inputs["attention_mask"].to(torch.int32)  # type: ignore
text_predictions = text_model.predict(
    {"input_ids": input_ids, "attention_mask": attention_mask}
)

# %%
cos_sim = np.dot(
    audio_predictions["projected_vec"].squeeze(),
    text_predictions["text_embedding"].squeeze(),
) / (
    np.linalg.norm(audio_predictions["projected_vec"].squeeze())
    * np.linalg.norm(text_predictions["text_embedding"].squeeze())
)

print("Similarity score:", cos_sim)

# %%
