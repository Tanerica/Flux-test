from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor 
import torch 
import gc 

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

ckpt_id = "black-forest-labs/FLUX.1-dev"
prompt = "a photo of a dog with cat-like look"

pipeline = FluxPipeline.from_pretrained(
    ckpt_id,
    transformer=None,
    vae=None,
    device_map="balanced",
    max_memory={0: "16GB", 1: "16GB"},
    torch_dtype=torch.bfloat16
)
print(pipeline.hf_device_map)

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

transformer = FluxTransformer2DModel.from_pretrained(
    ckpt_id, 
    subfolder="transformer",
    device_map="auto",
    max_memory={0: "16GB", 1: "16GB"},
    torch_dtype=torch.bfloat16
)
print(transformer.hf_device_map)
pipeline = FluxPipeline.from_pretrained(
    ckpt_id,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    vae=None,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

print("Running denoising.")
height, width = 768, 1360
# No need to wrap it up under `torch.no_grad()` as pipeline call method
# is already wrapped under that.
latents = pipeline(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=50,
    guidance_scale=3.5,
    height=height,
    width=width,
    output_type="latent",
).images
print(latents.shape)

del pipeline.transformer
del pipeline

flush()

vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

with torch.no_grad():
    print("Running decoding.")
    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    image[0].save("split_transformer.png")