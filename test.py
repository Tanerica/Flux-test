import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers import FluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize 
from huggingface_hub import login
login(token="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")
base_model = 'black-forest-labs/FLUX.1-dev'
# controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Canny'
# controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16).to('cuda')
print(torch.cuda.memory_allocated() / (1024**3))
transformer = FluxTransformer2DModel.from_pretrained(
    base_model, 
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
quantize(transformer, weights=qfloat8)
freeze(transformer)
transformer.to('cuda')
print(torch.cuda.memory_allocated() / (1024**3))