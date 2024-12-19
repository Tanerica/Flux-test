from transformers import T5EncoderModel
import time
import gc
import torch
import diffusers

def flush():
    gc.collect()
    torch.cuda.empty_cache()

t5_encoder = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", subfolder="text_encoder_2", revision="refs/pr/7", torch_dtype=torch.bfloat16
)
text_encoder = diffusers.DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    text_encoder_2=t5_encoder,
    transformer=None,
    vae=None,
    revision="refs/pr/7",
)
pipeline = diffusers.DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", 
    torch_dtype=torch.bfloat16,
    revision="refs/pr/1",
    text_encoder_2=None,
    text_encoder=None,
)
pipeline.enable_model_cpu_offload()

@torch.inference_mode()
def inference(self, prompt, num_inference_steps=4, guidance_scale=0.0, width=1024, height=1024):
    self.text_encoder.to("cuda")
    start = time.time()
    (
        prompt_embeds,
        pooled_prompt_embeds,
        _,
    ) = self.text_encoder.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)
    self.text_encoder.to("cpu")
    flush()
    print(f"Prompt encoding time: {time.time() - start}")
    output = self.pipeline(
        prompt_embeds=prompt_embeds.bfloat16(),
        pooled_prompt_embeds=pooled_prompt_embeds.bfloat16(),
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    image = output.images[0]
    return image