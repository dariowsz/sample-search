# %%
import research.start as start  # noqa isort:skip

# %%
import coremltools as ct
import laion_clap
import torch
import torch.nn.functional as F
from laion_clap.clap_module.model import trace_model

# %%
# Initialize model and encode audio
model = laion_clap.CLAP_Module(
    enable_fusion=False,
    amodel="HTSAT-base",
    tmodel="roberta",
)
model.load_ckpt("checkpoints/630k-audioset-fusion-best.pt")
model = model.eval()

# %%
example_input = torch.rand(1, 480000)
# example_input["longer"] = torch.tensor([True])
out = model.model.audio_branch(example_input)
out_proj = model.model.audio_projection(out["embedding"])
out_proj = F.normalize(out_proj, dim=-1)
out_proj.shape

# %%
# This is not working
tr_model = trace_model(model.model)
print(tr_model)

# %%
