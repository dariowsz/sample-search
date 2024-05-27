# TODO: Not working correctly!

# %%
import research.start as start  # noqa isort:skip

# %%
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
test_audio = ["/Users/dario.wisznewer/Test_Samples/Cymatics-Bongo4.wav"]
audio_inputs = model.preprocess_audio(test_audio, resample=True)

scripted_clap_model = torch.jit.script(model.clap)

scripted_clap_model.save("scripted_msclap.pt")  # type: ignore

output = scripted_clap_model(audio_inputs, text_inputs)  # type: ignore
# %%
