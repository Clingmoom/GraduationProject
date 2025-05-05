import torch
import time
import pickle
import numpy as np
import argparse, os

from transformers import GPT2Tokenizer
from .configs import get_configs, ROOT_DIR
from src.models import GPTActor
from src.trainers import PromptScorer


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    nargs="?",
    default=ROOT_DIR / "ckpt" / "PAE" / "actor_step3000.pt",
)
parser.add_argument(
    "--save",
    type=str,
    nargs="?",
    default="./result",
)
parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="A photo of a cat",
)
parser.add_argument(
    "--seed",
    type=int,
    nargs="?",
    default=42,
)
parser.add_argument(
    "--card",
    type=int,
    nargs="?",
    default=0,
)
parser.add_argument(
    "--data",
    type=str,
    nargs="?",
    default="coco",
)

opt_a = parser.parse_args()
torch.manual_seed(opt_a.seed)
sft = opt_a.ckpt
device = f"cuda:{opt_a.card}"
cfg = get_configs("gpt2-medium")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", device=device)

def prepare_gpt2_input(prompt, device):
    enc = tokenizer
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode

step_dict = {
    0: torch.tensor(tokenizer.encode("0-0.5"),device=device),#0-0.5
    1: torch.tensor(tokenizer.encode("0-1"),device=device), #0-1
    2: torch.tensor(tokenizer.encode("0.5-1"),device=device),#0.5-1
}
w_dict = {
    0: torch.tensor(tokenizer.encode("0.5"),device=device),
    1: torch.tensor(tokenizer.encode("0.75"),device=device),
    2: torch.tensor(tokenizer.encode("1.0"),device=device),
    3: torch.tensor(tokenizer.encode("1.25"),device=device),
    4: torch.tensor(tokenizer.encode("1.5"),device=device),
}
token_dict = {
    ",": torch.tensor(tokenizer.encode(",")[0],device=device),
    ".": torch.tensor(tokenizer.encode(".")[0],device=device),
    ":": torch.tensor(tokenizer.encode(":")[0],device=device),
    " [": torch.tensor(tokenizer.encode(" [")[0],device=device),
    "[": torch.tensor(tokenizer.encode("[")[0],device=device),
    "]": torch.tensor(tokenizer.encode("]")[0],device=device),
    " ": torch.tensor(tokenizer.encode(" ")[0],device=device)
}

pattern = r'\[([^]]*):0-1:1\.0\]'#r'\[(\s*\w+):0-1:1\.0\]'

def trans_token(bef_list, diffw_list, diffstep_list):
    if len(bef_list) == 0:
        return bef_list

    aft_list = torch.tensor([], device=device)
    ind = 0

    def get_modes(special_token_ind_list):
        try:
            w_counts = torch.bincount(diffw_list[special_token_ind_list])
            w_mode = int(torch.argmax(w_counts).item())
        except:
            w_mode = 2  # default to 1.0

        try:
            counts = torch.bincount(diffstep_list[special_token_ind_list])
            mode = int(torch.argmax(counts).item())
        except:
            mode = 1  # default to 0-1
        return w_mode, mode

    def insert_tokens(special_token_ind_list, w_mode, mode):
        nonlocal aft_list
        for ind in special_token_ind_list:
            s_token = bef_list[ind]
            parts = [
                token_dict["["].unsqueeze(0),
                s_token.unsqueeze(0),
                token_dict[":"].unsqueeze(0),
                step_dict[mode],
                token_dict[":"].unsqueeze(0),
                w_dict[w_mode],
                token_dict["]"].unsqueeze(0),
            ]
            aft_list = torch.cat([aft_list] + parts)

    def process_segment():
        nonlocal ind, aft_list
        special_token_ind_list = []

        # 收集 special tokens
        while ind < len(bef_list):
            token = bef_list[ind]
            if token in [token_dict[","], token_dict["."], token_dict[" "]] or tokenizer.decode([token.long()]).startswith(" "):
                break
            special_token_ind_list.append(ind)
            ind += 1

        w_mode, mode = get_modes(special_token_ind_list)
        insert_tokens(special_token_ind_list, w_mode, mode)

    while ind < len(bef_list):
        token = bef_list[ind]

        # 普通 token
        if token not in [token_dict[","], token_dict["."]]:
            aft_list = torch.cat([aft_list, token.unsqueeze(0)])
            ind += 1
        else:
            # 遇到 , 或 . 时，先跳过标点，收集后续 token 再处理
            punct_token = token
            ind += 1
            special_token_ind_list = []

            while ind < len(bef_list):
                token = bef_list[ind]
                if token in [token_dict[","], token_dict["."]]:
                    break
                special_token_ind_list.append(ind)
                ind += 1

            aft_list = torch.cat([aft_list, punct_token.unsqueeze(0)])

            w_mode, mode = get_modes(special_token_ind_list)
            insert_tokens(special_token_ind_list, w_mode, mode)

    return aft_list

def generate_gpt2(model, prompt, device):
    temperature = 0.9
    top_k = 200

    x, decode = prepare_gpt2_input(prompt, device)
    max_new_tokens = 75-x.shape[-1]
    y, diffw_list, diffstep_list = model.generate_dy(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k)
    if y.shape == torch.Size([0]):
        return prompt
    y_0=y[0].long()

    input_w = diffw_list[0].long()
    input_step = diffstep_list[0].long()

    target_value = torch.tensor(50256, device=device)

    end = (y_0 == target_value).nonzero(as_tuple = True)[0]
    if end.numel() > 0:
        y_0 = y_0[:end[0]]
        input_w=input_w[:end[0]]
        input_step=input_step[:end[0]]

    res = decode(torch.cat([x[0], trans_token(y_0, input_w, input_step)]))
    end = res.find("[<|endoftext|>")
    if end > 0:
        res= res[:end]
    end = res.find("<|endoftext|>")
    if end > 0:
        res=res[:end]

    return res

tic = time.time()
scorer = PromptScorer(device=device, num_images_per_prompt=1, seed=opt_a.seed)
bs = 5

if opt_a.data == "coco":
    filename = ROOT_DIR / "data" / "evaluate_data" / "COCO_test_1k.npy"
    data = np.load(filename)
elif opt_a.data == "diffusiondb_test":
    data = np.load(ROOT_DIR / "data" / "evaluate_data" / "diffusiondb_test_1k.npy")
    data = data.reshape(1000, 1)
elif opt_a.data == "lexica":
    data = np.load(ROOT_DIR / "data" / "evaluate_data" / "lexica_test_1k.npy")
    data = data.reshape(1000, 1)

prompt_all = []
aes_sum, clip_scores_sum, final_scores_sum = [torch.tensor(0.0) for i in range(3)]

save_path = opt_a.save
os.makedirs(save_path, exist_ok=True)

with torch.inference_mode():
    gpt_sft = torch.compile(GPTActor.from_checkpoint(cfg, sft)).to(device)
    gpt_sft.eval()
    for i in range(0, len(data), 25):
        if i + 25 < len(data):
            p = i + 25
        else:
            p = len(data)
        plain_texts = [s[0] for s in data[i:p]]
        prompt = [generate_gpt2(gpt_sft, s, device) for s in plain_texts]
        prompt_all += prompt
        if p > 998:
            print(prompt, i)
        try:
            images = scorer.gen_image_batched(prompt)
            image_features = scorer.get_clip_features(images, is_batched=True)
            aes_scores = scorer.get_aesthetic_score(image_features, is_batched=True)
            aes_sum += torch.Tensor(aes_scores).sum()

            clip_scores = scorer.get_clip_score_batched(image_features, plain_texts)
            clip_scores_sum += torch.Tensor(clip_scores).sum()

            save = [x for x in range(i, p)]
            [images[ii].save(
                    os.path.join(save_path, f"{save[ii]:05}.jpg")
                )for ii in range(len(images))
            ]
        except:
            print("error", prompt, i)
            exit()

print(opt_a.save, round(aes_sum.item() * 0.001, 2), round(clip_scores_sum.item() * 0.001, 2))
npy_path = opt_a.save
os.makedirs(npy_path, exist_ok=True)
np.save(os.path.join(npy_path, "prompt.npy"), np.array(prompt_all))

data_dict = {
    "aes": round(aes_sum.item() * 0.001, 2),
    "clip": round(clip_scores_sum.item() * 0.001, 2),
}

with open(os.path.join(npy_path, "data_dict.pickle"), "wb") as file:
    pickle.dump(data_dict, file)

toc = time.time()
print(f"time:{toc - tic}")