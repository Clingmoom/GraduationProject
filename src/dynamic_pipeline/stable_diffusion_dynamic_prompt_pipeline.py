import torch

from diffusers import StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin,TextualInversionLoaderMixin
from typing import *
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from special_token import SpecialToken


def delete_elements(
    list_input: List[str], indexes: List[int]
) -> Tuple[List[str], Dict[int, Optional[int]]]:
    """
    Remove elements at specified indexes from a token list.

    Args:
        list_input: Original list of token strings.
        indexes: Positions to remove.
    Returns:
        new_list: Filtered list with specified positions removed.
        index_mapping: Mapping from original indexes to new indexes (or None if removed).
    """
    keep_set = set(indexes)
    # Build new list without in-place deletions
    new_list = [tok for i, tok in enumerate(list_input) if i not in keep_set]
    # Build mapping from old index -> new index or None
    index_mapping: Dict[int, Optional[int]] = {}
    offset = 0
    for i in range(len(list_input)):
        if i in keep_set:
            index_mapping[i] = None
            offset += 1
        else:
            index_mapping[i] = i - offset
    return new_list, index_mapping


class StableDiffusionDynamicPromptPipeline(StableDiffusionPipeline):
    """
    Extends StableDiffusionPipeline to support dynamic prompt weights.

    Special syntax in prompt: [text:startFrac-endFrac:weight]
    will apply 'weight' to 'text' between specified denoising steps.
    """

    def _extract_special_tokens(
        self, prompt: str, num_steps: int
    ) -> List[SpecialToken]:
        """
        Scan the raw prompt to extract SpecialToken objects.

        Args:
            prompt: Raw prompt string containing tags.
            num_steps: Total denoising steps.
        Returns:
            List of SpecialToken instances.
        """
        special_tokens: List[SpecialToken] = []
        clean_acc = ""
        i = 0
        while i < len(prompt):
            if prompt[i] != "[":
                clean_acc += prompt[i]
                i += 1
                continue
            # Found a '[', parse tag
            start = i + 1
            while i < len(prompt) and prompt[i] != "]":
                i += 1
            raw = prompt[start:i]
            parts = raw.split(":")
            if len(parts) == 3:
                text, range_raw, weight_raw = parts
                try:
                    start_frac, end_frac = map(float, range_raw.split("-"))
                    start_step = int(start_frac * num_steps)
                    end_step = int(end_frac * num_steps)
                    steps = list(range(start_step, end_step + 1))
                    weight = float(weight_raw)
                    subtokens = self.tokenizer.tokenize(text)
                    base_idx = len(self.tokenizer.tokenize(clean_acc))
                    for j, tok in enumerate(subtokens):
                        tok_str = self.tokenizer.convert_tokens_to_string(tok)
                        special_tokens.append(
                            SpecialToken(tok_str, base_idx + j, steps, weight)
                        )
                except Exception:
                    clean_acc += f"[{raw}]"
            else:
                clean_acc += f"[{raw}]"
            i += 1
        return special_tokens

    def _merge_subtokens(
        self, special_tokens: List[SpecialToken], tokenized: List[str]
    ) -> List[SpecialToken]:
        """
        Optionally merge adjacent subtokens if tokenization splits words.

        Args:
            special_tokens: Extracted subtokens.
            tokenized: Full token list of the prompt.
        Returns:
            Possibly merged list of SpecialToken.
        """
        # For simplicity, assume no merging needed
        return special_tokens

    def _build_clean_prompts(
        self,
        prompt: str,
        special_tokens: List[SpecialToken],
        num_steps: int,
    ) -> Tuple[Dict[int, Tuple[str, List[Tuple[float, int]]]], set]:
        """
        Construct per-step clean prompts and weight-index mappings.

        Args:
            prompt: Original prompt string.
            special_tokens: Extracted SpecialToken list.
            num_steps: Total denoising steps.
        Returns:
            prompt_map: step -> (clean_prompt, [(weight, index), ...])
            unique_prompts: set of clean_prompt variations
        """
        clean_text = prompt
        all_toks = self.tokenizer.tokenize(clean_text)
        prompt_map: Dict[int, Tuple[str, List[Tuple[float, int]]]] = {}
        unique_prompts = set()
        for step in range(num_steps):
            toks = all_toks.copy()
            active: List[Tuple[float, int]] = []
            remove_idxs: List[int] = []
            for sp in special_tokens:
                if step in sp.steps:
                    active.append((sp.weight, sp.index_in_all_tokens))
                else:
                    remove_idxs.append(sp.index_in_all_tokens)
            toks, idx_map = delete_elements(toks, remove_idxs)
            remapped = [(w, idx_map[i]) for w, i in active if idx_map.get(i) is not None]
            clean_str = self.tokenizer.convert_tokens_to_string(toks)
            prompt_map[step] = (clean_str, remapped)
            unique_prompts.add(clean_str)
        return prompt_map, unique_prompts

    def parse_single_prompt(
        self, prompt: str, num_inference_steps: int
    ) -> Tuple[Dict[int, Tuple[str, List[Tuple[float, int]]]], set]:
        """
        Parse a prompt for dynamic weights across inference steps.

        Args:
            prompt: Prompt string with [text:start-end:weight] syntax.
            num_inference_steps: Total denoising steps.
        Returns:
            A mapping per step to clean prompt and (weight,index) list,
            plus the set of unique clean prompts.
        """
        # Normalize spacing
        prompt = prompt.replace("[ ", "[")
        # Extract and preprocess tokens
        sp_tokens = self._extract_special_tokens(prompt, num_inference_steps)
        sp_tokens = self._merge_subtokens(sp_tokens, self.tokenizer.tokenize(prompt))
        # Build clean prompts and mappings
        return self._build_clean_prompts(prompt, sp_tokens, num_inference_steps)

    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            num_inference_steps=50
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # 检查提示词类型
        assert isinstance(prompt, str)
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        ### Prompt parsing by Terry Zhang
        # step 1: find all special tokens that has [] surrounded, find their indexes after tokenizing
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        prompt_raw = prompt
        clean_prompt_and_specialtoken_weightindex_pair, clean_prompt_set = self.parse_single_prompt(prompt_raw,
                                                                                                    num_inference_steps)
        # 编码每个干净提示词
        clean_prompt_to_prompt_embeds = dict()
        for clean_prompt in clean_prompt_set:
            text_inputs = self.tokenizer(
                clean_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                # print(
                #     "The following part of your input was truncated because CLIP can only handle sequences up to"
                #     f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                # )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

            if self.text_encoder is not None:
                prompt_embeds_dtype = self.text_encoder.dtype
            elif self.unet is not None:
                prompt_embeds_dtype = self.unet.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            clean_prompt_to_prompt_embeds[clean_prompt] = prompt_embeds

        # 处理无条件嵌入
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # 生成每个步骤的提示词嵌入
        prompt_embeds_dict = {}
        for step in clean_prompt_and_specialtoken_weightindex_pair.keys():
            clean_prompt_of_step, sp_list_of_step = clean_prompt_and_specialtoken_weightindex_pair[step]
            prompt_embeds_of_step = clean_prompt_to_prompt_embeds[clean_prompt_of_step]
            weight = torch.ones_like(prompt_embeds_of_step)

            for w, ind in sp_list_of_step:
                if ind != None and ind + 1 < weight.size(1):
                    weight[:, ind + 1] = w
            previous_mean = prompt_embeds_of_step.float().mean(axis=[-2, -1]).to(prompt_embeds_of_step.dtype)
            prompt_embeds_dict[step] = prompt_embeds_of_step * weight
            current_mean = prompt_embeds_dict[step].float().mean(axis=[-2, -1]).to(prompt_embeds_dict[step].dtype)
            # prompt_embeds_dict[step]  *= (previous_mean / current_mean)
            expanded_previous_mean = previous_mean.unsqueeze(-1).unsqueeze(-1)
            expanded_current_mean = current_mean.unsqueeze(-1).unsqueeze(-1)
            expanded_previous_mean = expanded_previous_mean.expand_as(prompt_embeds_dict[step])
            expanded_current_mean = expanded_current_mean.expand_as(prompt_embeds_dict[step])

            prompt_embeds_dict[step] *= (expanded_previous_mean / expanded_current_mean)

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                prompt_embeds_dict[step] = torch.cat([negative_prompt_embeds, prompt_embeds_dict[step]])
            # print(step,clean_prompt_of_step,weight[0,:,0])
        return prompt_embeds_dict

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        assert prompt_embeds is None
        assert negative_prompt_embeds is None
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        if isinstance(prompt, list):
            prompt_embeds_dict = None
            for prompt_str in prompt:
                prompt_embeds_dict_one = self._encode_prompt(
                    prompt_str,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                    num_inference_steps=len(timesteps)
                )
                if prompt_embeds_dict is None:
                    prompt_embeds_dict = prompt_embeds_dict_one
                else:
                    for k in prompt_embeds_dict.keys():
                        new_v = prompt_embeds_dict_one[k]

                        if do_classifier_free_guidance:
                            neg_p, pos_p = torch.chunk(new_v, 2)
                            neg_p_, pos_p_ = torch.chunk(prompt_embeds_dict[k], 2)
                            neg_p_cat = torch.cat([neg_p_, neg_p])
                            pos_p_cat = torch.cat([pos_p_, pos_p])
                            prompt_embeds_dict[k] = torch.cat([neg_p_cat, pos_p_cat])
                        else:
                            prompt_embeds_dict[k] = torch.cat([prompt_embeds_dict[k], new_v])

        else:
            prompt_embeds_dict = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                num_inference_steps=len(timesteps)
            )

        # 4. Prepare timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds_dict[0].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_dict[i],
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds_dict[0].dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)