import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler, MultiControlNetModel, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
from utils.depth_utils import estimate_depth
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2

# Load ControlNet for depth
depth_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16
)

# Load ControlNet for Canny Edge
canny_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

multi_controlnet = MultiControlNetModel([depth_controlnet, canny_controlnet])

# Load base Stable Diffusion model
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-1-base",
#     controlnet=controlnet,
#     torch_dtype=torch.float16
# ).to("cuda")
 
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    # "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    # "SG161222/Realistic_Vision_V5.1_noVAE", #good
    "lykon/dreamshaper-8",
    controlnet=multi_controlnet, 
    torch_dtype=torch.float16
).to("cuda")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipe.vae = vae
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
transform = transforms.Compose([transforms.PILToTensor()])

# Load your images
init_image = load_image("/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/fern/test/ours_3000/renders/IMG_4026.png").convert("RGB")
# init_image = load_image("/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/trex/test/ours_3000/renders/DJI_20200223_163557_660.png").convert("RGB")
# init_image = load_image("/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/orchids/test/ours_3000/renders/IMG_4483.png").convert("RGB")
inp_img = transform(init_image).cuda()
control_image = estimate_depth(inp_img,mode='train')#load_image("depth.png").convert("RGB")

print(init_image.size)
# Optional: Resize to 512px if needed
init_image = init_image.resize((512, 512))
# control_image = control_image.resize((512, 512))
control_image = F.interpolate(control_image.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0).detach().cpu()#.convert("RGB")
transform = transforms.ToPILImage()
control_image = transform(control_image).convert('RGB')

low_threshold = 100
high_threshold = 200
image = np.array(init_image)
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)


# Run the pipeline
# result = pipe(
#     prompt="a photo of the same scene, same lighting condition, consistent with depth, high realism, sharp boundary, without blur",#"photo-realistic interior, same lighting, consistent with depth",
#     negative_prompt="blurry, over-saturated, distorted, different objects",
#     image=init_image,
#     control_image=image,
#     num_inference_steps=10,
#     strength=1,  # How much it can change your init_image
#     guidance_scale=6
# )

result = pipe(
    # prompt="a realistic photo of the same scene, sharp focus, same lighting, consistent with depth",
    # negative_prompt="distorted, hallucinated objects, blur, fantasy, surreal",
    prompt = (
        "sharp focus, same lighting, clean background, consistent with original depth and layout"
    ),
    negative_prompt = (
        "blurry, distorted, surreal, fantasy, overexposed, different object, cartoon"
    ),
    image=init_image,
    control_image=[control_image,image],
    num_inference_steps=30,
    strength=0.15,   # allows image to be refined, not replaced
    guidance_scale=10
)
result.images[0].save("refined_view.png")
init_image.save("input1.png")