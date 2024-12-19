from huggingface_hub import login
login(token="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")
import torch
from diffusers import FluxPipeline
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# prompt = "A cat holding a sign that says hello world"
# out = pipe(
#     prompt=prompt,
#     guidance_scale=0.,
#     height=768,
#     width=1360,
#     num_inference_steps=4,
#     max_sequence_length=256,
# ).images[0]
# out.save("imageschnell.png")
prompt = "a tiny astronaut hatching from an egg on the moon"
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
).images[0]
out.save("image50.png")