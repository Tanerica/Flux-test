from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor 
import torch 
import gc 
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from image_gen_aux import DepthPreprocessor
from optimum.quanto import freeze, qfloat8, quantize
from huggingface_hub import login
login(token="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")
def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

ckpt_id = "black-forest-labs/FLUX.1-dev"
prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."

pipeline = FluxPipeline.from_pretrained(
    ckpt_id,
    transformer=None,
    vae=None,
    torch_dtype=torch.bfloat16
).to('cuda')

with torch.no_grad():
    print("Encoding prompts.")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=512
    )

print(prompt_embeds.shape)

del pipeline.text_encoder
del pipeline.text_encoder_2
del pipeline.tokenizer
del pipeline.tokenizer_2
del pipeline
flush()
print('C0: ', torch.cuda.memory_allocated() / (1024 ** 3))

from diffusers.utils import load_image
flush()
controlnet_depth = FluxControlNetModel.from_pretrained('InstantX/FLUX.1-dev-Controlnet-Union', torch_dtype=torch.bfloat16)

control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")
control_image.save('depth.png')

transformer = FluxTransformer2DModel.from_pretrained(
    ckpt_id, 
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
quantize(transformer, weights=qfloat8)
freeze(transformer)

pipeline = FluxControlNetPipeline.from_pretrained(
    ckpt_id,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    controlnet=controlnet_depth, 
    transformer=transformer,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

print("Running denoising.")

image = pipeline(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    control_mode=2,
    num_inference_steps=20,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    guidance_scale=3.5,
    output_type="pil",
).images[0]
image.save('control_depth.png')

del pipeline.transformer
del pipeline
del transformer
del controlnet_depth
flush()
print('B0: ',torch.cuda.memory_allocated() / (1024 ** 3))