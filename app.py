import gradio as gr
import spaces
from diffusers import StableDiffusionXLPipeline
import torch
import random
from PIL import Image

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Poppins'), 'Arial', 'sans-serif']
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    custom_pipeline="multimodalart/sdxl_perturbed_attention_guidance",
    torch_dtype=torch.float16
)

device = "cuda"
pipe = pipe.to(device)

@spaces.GPU
def generate_images(prompt, negative_prompt=None, guidance_scale=7.0, pag_scale=3.0, pag_layers=["mid"], randomize_seed=True, seed=42, lora=None, progress=gr.Progress(track_tqdm=True)):
    """Generate images using Stable Diffusion XL with and without Perturbed-Attention Guidance."""
    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip() if negative_prompt and negative_prompt.strip() else None

    if randomize_seed:
        seed = random.randint(0, 9007199254740991)

    if not prompt and not negative_prompt:
        guidance_scale = 0.0

    pipe.unfuse_lora()
    pipe.unload_lora_weights()

    if lora:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora(lora_scale=0.9)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image_pag = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
                     pag_scale=pag_scale, pag_applied_layers=pag_layers, generator=generator, num_inference_steps=25).images[0]

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image_normal = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
                        generator=generator, num_inference_steps=25).images[0]

    return [image_pag, image_normal], seed

css = '''
.gradio-container {
    max-width: 800px !important;
    margin: 0 auto;
    font-family: 'Poppins', Arial, sans-serif;
    background-color: #f9fafb;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
h1 {
    text-align: center;
    color: #4f46e5;
    font-weight: bold;
}
p {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
}
'''

with gr.Blocks(css=css, theme=theme) as demo:
    gr.Markdown("""
    # **Creative Image Generation with PAG**  
    Generate stunning visuals with or without Perturbed-Attention Guidance (PAG) using the state-of-the-art **Stable Diffusion XL**. 
    """)

    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(
                show_label=False,
                placeholder="Enter your creative idea here...",
                scale=4
            )
            button = gr.Button("Create", variant="primary")

        output = gr.Gallery(
            label="Comparison: Left (PAG) | Right (No PAG)",
            columns=2
        )

        with gr.Accordion("Advanced Settings", open=False):
            guidance_scale = gr.Number(
                label="CFG Guidance Scale",
                value=7.0
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Avoid specific details here..."
            )
            pag_scale = gr.Number(
                label="PAG Scale",
                value=3.0
            )
            pag_layers = gr.Dropdown(
                label="PAG Layers",
                choices=["up", "mid", "down"],
                multiselect=True,
                value="mid"
            )
            randomize_seed = gr.Checkbox(
                label="Randomize Seed",
                value=True
            )
            seed = gr.Slider(
                label="Seed",
                minimum=1,
                maximum=9007199254740991,
                step=1
            )
            lora = gr.Textbox(
                label="Custom LoRA Path",
                placeholder="Optional: Provide a custom LoRA model path"
            )

    gr.Examples(
        examples=[
            ["a futuristic cityscape under a vibrant sunset"],
            ["a magical forest with glowing mushrooms, fantasy art"],
            ["a robot artist painting a surreal masterpiece"]
        ],
        inputs=[prompt],
        outputs=[output, seed],
        fn=generate_images,
        cache_examples=False
    )

    button.click(
        fn=generate_images,
        inputs=[prompt, negative_prompt, guidance_scale, pag_scale, pag_layers, randomize_seed, seed, lora],
        outputs=[output, seed]
    )

if __name__ == "__main__":
    demo.launch(share=True)
