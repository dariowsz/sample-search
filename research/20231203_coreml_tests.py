# %%
import research.start as start  # noqa isort:skip

# %%
import coremltools as ct
import torch
import torchvision

# %%
# Load a pre-trained version of MobileNetV2
torch_model = torchvision.models.mobilenet_v2(pretrained=True)
# Set the model in evaluation mode.
torch_model.eval()

# Trace the model with random data.
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)  # type: ignore

# %%
# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
)

# %%
# Save the converted model.
model.save("newmodel.mlpackage")  # type: ignore

# %%
