import os
import torch
import argparse
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from datetime import datetime
from utils.lora_utils import train_lora
import imageio
from main import main
LENGTH=450

def train_lora_interface(
    image0,
    image1,
    prompt,
    model_path,
    vae_path,
    lora_path,
    lora_steps,
    lora_rank,
    lora_lr,
    progress=gr.Progress()
):
    os.makedirs(lora_path, exist_ok=True)
    train_lora(image0, image1, prompt, model_path, vae_path, lora_path,
               lora_steps, lora_lr, lora_rank, progress)
    return f"Train LoRA Done!"

with gr.Blocks() as demo:
    
    with gr.Row():
        gr.Markdown("""
        # Official Implementation of [DreamMover](https://github.com/leoShen917/DreamMover)
        """)

    original_image_0, original_image_1 = gr.State(Image.open("assets/dog0.png").convert("RGB")), gr.State(Image.open("assets/dog1.png").convert("RGB"))
    # key_points_0, key_points_1 = gr.State([]), gr.State([])
    # to_change_points = gr.State([])
    
    with gr.Row():
        with gr.Column():
            input_img_0 = gr.Image(type="numpy", label="Input image 0", value="assets/dog0.png", show_label=True, height=LENGTH, width=LENGTH, interactive=True)
        with gr.Column():
            input_img_1 = gr.Image(type="numpy", label="Input image 1 ", value="assets/dog1.png", show_label=True, height=LENGTH, width=LENGTH, interactive=True)
        with gr.Column():
            output_video = gr.Video(format="mp4", label="Output video", show_label=True, height=LENGTH, width=LENGTH, interactive=False)
    # general parameters
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path")
        lora_status_bar = gr.Textbox(label="display LoRA training status")

    with gr.Row():
        with gr.Column():
            train_lora_button = gr.Button("Train LoRA")
        with gr.Column():
            run_button = gr.Button("Run")
        with gr.Column():
            clear_button = gr.Button("Clear All")    
    
    with gr.Accordion(label="Algorithm Parameters"):
        with gr.Tab("Basic Settings"):
            with gr.Row():
                model_path = gr.Text(value='runwayml/stable-diffusion-v1-5',
                    label="Diffusion Model Path", interactive=True
                )
                vae_path = gr.Text(value="default",
                    label="VAE Model Path", interactive=True
                )
            with gr.Row():
                num_frames = gr.Number(value=33, minimum=0, label="Number of Frames", precision=0, interactive=True)
                fps = gr.Number(value=10, minimum=0, label="FPS (Frame rate)", precision=0, interactive=True)
                save_inter = gr.Checkbox(value=True, label="Save Intermediate Images", interactive=True)
                output_path = gr.Text(value="./results", label="Output Path", interactive=True)
                
        with gr.Tab("LoRA Settings"):
            with gr.Row():
                lora_steps = gr.Number(value=200, label="LoRA training steps", precision=0, interactive=True)
                lora_lr = gr.Number(value=0.0002, label="LoRA learning rate", interactive=True)
                lora_rank = gr.Number(value=16, label="LoRA rank", precision=0, interactive=True)

        with gr.Tab("Diffusion Settings"):
            with gr.Row():
                guidance_scale = gr.Number(value=1, label="CFG", precision=0, interactive=False)
                n_inference_step = gr.Number(value=50, label="total noisy steps", precision=0, interactive=False)
                n_actual_inference_step = gr.Number(value=30, label="actual inversion steps", precision=0, interactive=True)
                feature_inversion = gr.Number(value=14, label="the noisy steps for flow", precision=0, interactive=True)
                unet_feature_idx = gr.Number(value=2, label="which up-blocks for flow", precision=0, interactive=False)
                
    def store_img(img):
        image = Image.fromarray(img).convert("RGB")
        # resize the input to 512x512
        # image = image.resize((512,512), Image.BILINEAR)
        # image = np.array(image)
        # when new image is uploaded, `selected_points` should be empty
        return image
    input_img_0.upload(
        store_img,
        [input_img_0],
        [original_image_0]
    )
    input_img_1.upload(
        store_img,
        [input_img_1],
        [original_image_1]
    )

    train_lora_button.click(
        train_lora_interface,
        [
         original_image_0,
         original_image_1,
         prompt,
         model_path,
         vae_path,
         lora_path,
         lora_steps,
         lora_rank,
         lora_lr
        ],
        [lora_status_bar]
    )

    def clear(LENGTH):
        return gr.update(value=None, width=LENGTH, height=LENGTH), \
            gr.update(value=None, width=LENGTH, height=LENGTH), \
            gr.update(value=None, width=LENGTH, height=LENGTH), \
            None, None, None
    clear_button.click(
        clear,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [input_img_0, input_img_1, output_video, original_image_0, original_image_1, prompt]
    )
    def run_mover(
        prompt,
        img0,
        img1,
        model_path,
        vae_path,
        lora_path,
        save_dir,
        guidance_scale,
        n_inference_step,
        feature_inversion,
        n_actual_inference_step,
        unet_feature_idx,
        Time,
        fps,
        save_inter
    ):
        parser = argparse.ArgumentParser(description='frame interpolator model')
        args = parser.parse_args()
        args.prompt = prompt
        image = np.stack((np.array(img0),np.array(img1)))
        source_image =  torch.from_numpy(image).float() / 127.5 - 1
        args.source_image = source_image.permute(0,3,1,2)
        args.model_path = model_path
        args.vae_path = vae_path
        args.lora_path = lora_path
        args.save_dir = save_dir
        args.guidance_scale = guidance_scale
        args.n_inference_step = n_inference_step
        args.feature_inversion = feature_inversion
        args.n_actual_inference_step = n_actual_inference_step
        args.unet_feature_idx = []
        args.unet_feature_idx.append(unet_feature_idx)
        args.Time = Time
        args.save_inter = save_inter
        
        gen_image = main(args)

        gen_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()
        gen_image = (gen_image * 255).astype(np.uint8)
        video_path = os.path.join(args.save_dir, 'output.mp4')
        img_array=[]
        for filename in gen_image:
            img_array.append(filename)
        imageio.mimwrite(video_path, img_array, fps=fps, quality=8)
        return gr.Video(value=video_path, format="mp4", label="Output video", show_label=True, height=LENGTH, width=LENGTH, interactive=False)
    
    run_button.click(
        run_mover,
        [
        prompt,
        original_image_0,
        original_image_1,
        model_path,
        vae_path,
        lora_path,
        output_path,
        guidance_scale,
        n_inference_step,
        feature_inversion,
        n_actual_inference_step,
        unet_feature_idx,
        num_frames,
        fps,
        save_inter
        ],
        [output_video]
    )
    
demo.queue().launch(debug=True)