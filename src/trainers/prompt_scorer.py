import os
import torch
import random
import numpy as np
import clip

from torch import autocast
from diffusers import UniPCMultistepScheduler
from transformers import AutoProcessor, AutoModel
from .aesthetic_mlp import AestheticMlp
from src.dynamic_pipeline import StableDiffusionDynamicPromptPipeline
from ..configs import ROOT_DIR


class PromptScorer:
    def __init__(self , device, num_images_per_prompt=2, seed=None):
        # init scorer hparams

        self.num_images_per_prompt = num_images_per_prompt
        self.seed =seed
        # init models
        self.device = device
        self.init_clip_model()
        self.init_aesthetic_model()
        self.init_pickscore_model()
        self.init_diffusion_model()
        self.eval_data_res = []

    def init_pickscore_model(self):
        self.pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(self.device)

    def init_clip_model(self):
        # wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
        self.clip_model, self.clip_preprocess = clip.load(ROOT_DIR / "ckpt" / "CLIP_ViT" / "ViT-L-14.pt", device=self.device, jit=False)

    def init_aesthetic_model(self):
        model = AestheticMlp(768)
        s = torch.load(ROOT_DIR / "ckpt" / "aesthetic" / "sac+logos+ava1-l14-linearMSE.pth")
        model.load_state_dict(s)
        model.to(self.device)
        model.eval()
        self.aes_model = model

    def init_diffusion_model(self):
        device = self.device

        self.sdmodel_name= "CompVis/stable-diffusion-v1-4"
        dpm_scheduler = UniPCMultistepScheduler.from_pretrained(
            self.sdmodel_name, subfolder="scheduler"
        )

        pipe = StableDiffusionDynamicPromptPipeline.from_pretrained(
            self.sdmodel_name,
            variant="fp16",
            torch_dtype=torch.float16,
            scheduler=dpm_scheduler,
        )
        # Disable NSFW detect
        pipe.safety_checker = None
        pipe.feature_extractor = None

        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
        self.diffusion_pipe = pipe


    def get_pick_score(self, prompt, images):
        # device = "cuda:7"
        # preprocess
        if len(images) != len(prompt):
            assert len(images) % len(prompt) == 0
            copied_strings = []
            for pmt in prompt:
                copied_strings.extend([pmt] * 3)
            prompt = copied_strings

        image_inputs = self.pick_processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.pick_processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # embed
            image_embs = self.pick_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.pick_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores = self.pick_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        return scores.cpu().tolist()

    def get_clip_features(self, pil_image, is_batched=False):
        if not is_batched:
            image = self.clip_preprocess(pil_image).unsqueeze(0)
        else:
            images = [self.clip_preprocess(i) for i in pil_image]
            image = torch.stack(images)

        image = image.to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features

    def get_clip_score(self, image_features, prompt):
        tokens = clip.tokenize([prompt], truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logit = image_features @ text_features.t()
            score = logit.item()
        return score

    def get_clip_score_batched(self, image_features, prompts):
        tokens = clip.tokenize(prompts, truncate=True).to(self.device)

        with torch.no_grad():
            if len(image_features) != len(prompts):
                assert len(image_features) % len(prompts) == 0
                tokens = (
                    tokens.unsqueeze(1)
                    .expand(-1, self.num_images_per_prompt, -1)
                    .reshape(-1, tokens.shape[-1])
                )

            text_features = self.clip_model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logit = image_features @ text_features.t()
        scores = logit.diag().tolist()
        return scores

    def get_aesthetic_score(self, image_features, is_batched=False):
        features = image_features.cpu().detach().numpy()
        order = 2
        axis = -1
        l2 = np.atleast_1d(np.linalg.norm(features, order, axis))
        l2[l2 == 0] = 1
        im_emb_arr = features / np.expand_dims(l2, axis)
        prediction = self.aes_model(
            torch.from_numpy(im_emb_arr)
            .to(self.device)
            .type(torch.cuda.FloatTensor)
        )
        if is_batched:
            return prediction[:, 0].tolist()
        else:
            return prediction.item()

    def gen_image_batched(self, prompts, save=None, save_path="./image"):
        images = []
        bsz = 8
        if self.seed != None:
            for i in range(0, len(prompts), bsz):
                pmpts = prompts[i : i + bsz]
                with autocast("cuda"):
                    sub_images = self.diffusion_pipe(
                        pmpts,
                        num_images_per_prompt=self.num_images_per_prompt,
                        num_inference_steps=10,
                        generator=torch.Generator().manual_seed(int(self.seed))
                    ).images
                    images.extend(sub_images)
        else:
            for i in range(0, len(prompts), bsz):
                pmpts = prompts[i : i + bsz]
                try:
                    with autocast("cuda"):
                        sub_images = self.diffusion_pipe(
                            pmpts,
                            num_images_per_prompt=self.num_images_per_prompt,
                            num_inference_steps=10,
                        ).images
                        images.extend(sub_images)
                except:
                    print("!!!" ,pmpts)
                    exit()
        if save != None:
            os.makedirs(save_path ,exist_ok=True)
            [images[i].save(os.path.join(save_path , f'{save[i]:05}.png')) for i in range(len(images))]
        return images

    def get_score_batched(self, prompts, plain_texts, plain_aes_score=None):

        images = self.gen_image_batched(prompts)
        image_features = self.get_clip_features(images, is_batched=True)

        aes_scores = self.get_aesthetic_score(image_features, is_batched=True)
        aes_scores = torch.Tensor(aes_scores)

        clip_scores = self.get_clip_score_batched(image_features, plain_texts)
        clip_scores = torch.Tensor(clip_scores)
        clip_scores = torch.maximum(clip_scores, torch.zeros_like(clip_scores))

        pick_score = self.get_pick_score(plain_texts ,images)
        pick_score = torch.Tensor(pick_score)

        final_scores =  aes_scores + torch.where(clip_scores > 0.28, 0, 20 * clip_scores - 5.6) +torch.where(
            pick_score > 18, 0, pick_score - 18)

        if random.random() < 0.001:
            print(f"prompt:{prompts}")
            print(f"final_scores:{final_scores}")

        final_scores = final_scores.reshape(-1, self.num_images_per_prompt).mean(1).to(self.device)

        return final_scores