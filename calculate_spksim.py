import argparse
import io
import json
import os

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["FAIRSEQ2_CACHE_DIR"] = "/mnt/nvme1/fairseq2_cache"

from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset, Audio
from tqdm import tqdm


def get_single_prompt(gen_dialect):
    if gen_dialect in ["ALG", "IRQ", "UAE"]:
        ref_audio = str(files("habibi_tts").joinpath(f"assets/{gen_dialect}.wav"))
    elif gen_dialect in ["EGY", "MAR", "MSA"]:
        ref_audio = str(files("habibi_tts").joinpath(f"assets/{gen_dialect}.mp3"))
    elif gen_dialect[:3] == "SAU":
        ref_audio = str(files("habibi_tts").joinpath(f"assets/{gen_dialect[4:]}.wav"))
    else:
        ref_audio = str(files("habibi_tts").joinpath("assets/MSA.mp3"))

    if gen_dialect == "MSA":
        ref_text = "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."
    elif gen_dialect[:3] == "SAU":
        if gen_dialect[4:] == "Najdi":
            ref_text = "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه."
        elif gen_dialect[4:] == "Hijazi":
            ref_text = "ابغاك تحقق معاه بس بشكل ودي لانه سلطان يمر بظروف صعبة شوية."
        elif gen_dialect[4:] == "Gulf":
            ref_text = "وين تو الناس متى تصحى ومتى تفطر وتغير يبيلك ساعة يعني بالله تروح الشغل الساعة عشره."
        else:
            raise ValueError(f"[Code utils_eval.py] unexpected gen_dialect: {gen_dialect}")
    elif gen_dialect == "UAE":
        ref_text = "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا وحياناً هذا الستايل ما نحبه."
    elif gen_dialect == "ALG":
        ref_text = "أنيا هكا باغية ناكل هكا أني ن نشوف فيها الحاجة هذيكا."
    elif gen_dialect == "IRQ":
        ref_text = "يعني ااا ما نقدر ناخذ وقت أكثر، ااا لأنه شروط كلش يحتاجلها وقت."
    elif gen_dialect == "EGY":
        ref_text = "ايه الكلام. بقولك ايه. استخدم صوتي في المحادثات. استخدمه هيعجبك اوي."
    elif gen_dialect == "MAR":
        ref_text = "إذا بغيتي شي صوت باللهجة المغربية للإعلانات ديالك هذا أحسن واحد غادي تلقاه."
    else:
        ref_text = "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."

    return ref_audio, ref_text

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        channels=512,
        emb_dim=192,
        global_context_att=False,
        feat_type="wavlm_large",
        sr=16000,
        feature_selection="hidden_states",
        update_extract=False,
        config_path=None,
    ):
        super().__init__()

        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        try:
            local_s3prl_path = os.path.expanduser("~/.cache/torch/hub/s3prl_s3prl_main")
            self.feature_extract = torch.hub.load(local_s3prl_path, feat_type, source="local", config_path=config_path)
        except:  # noqa: E722
            self.feature_extract = torch.hub.load("s3prl/s3prl", feat_type)

        if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
            self.feature_extract.model.encoder.layers[23].self_attn, "fp32_attention"
        ):
            self.feature_extract.model.encoder.layers[23].self_attn.fp32_attention = False
        if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
            self.feature_extract.model.encoder.layers[11].self_attn, "fp32_attention"
        ):
            self.feature_extract.model.encoder.layers[11].self_attn.fp32_attention = False

        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type != "fbank" and feat_type != "mfcc":
            freeze_list = ["final_proj", "label_embs_concat", "mask_emb", "project_q", "quantizer"]
            for name, param in self.feature_extract.named_parameters():
                for freeze_val in freeze_list:
                    if freeze_val in name:
                        param.requires_grad = False
                        break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        # self.channels = [channels] * 4 + [channels * 3]
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            self.channels[0],
            self.channels[1],
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer3 = SE_Res2Block(
            self.channels[1],
            self.channels[2],
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer4 = SE_Res2Block(
            self.channels[2],
            self.channels[3],
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
            se_bottleneck_dim=128,
        )

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(
            self.channels[-1], attention_channels=128, global_context_att=global_context_att
        )
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x):
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                if self.feat_type == "fbank" or self.feat_type == "mfcc":
                    x = self.feature_extract(x) + 1e-6  # B x feat_dim x time_len
                else:
                    x = self.feature_extract([sample for sample in x])

        if self.feat_type == "fbank":
            x = x.log()

        if self.feat_type != "fbank" and self.feat_type != "mfcc":
            x = x[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)
            else:
                x = x.unsqueeze(0)
            norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        return out


def ECAPA_TDNN_SMALL(
    feat_dim,
    emb_dim=256,
    feat_type="wavlm_large",
    sr=16000,
    feature_selection="hidden_states",
    update_extract=False,
    config_path=None,
):
    return ECAPA_TDNN(
        feat_dim=feat_dim,
        channels=512,
        emb_dim=emb_dim,
        feat_type=feat_type,
        sr=sr,
        feature_selection=feature_selection,
        update_extract=update_extract,
        config_path=config_path,
    )


def calculate_spksim(wav_dir, dialect, ckpt, single):
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    model = model.to(device)
    model.eval()

    benchmark = load_dataset("SWivid/Habibi", dialect, split="test")
    benchmark = benchmark.cast_column("audio", Audio(decode=False))

    # Build speaker-to-indices map for cross-comparison
    spk_to_indices = defaultdict(list)
    for idx, obj in enumerate(benchmark):
        spk_to_indices[obj["speaker_id"]].append(idx)

    spksim_objs = []
    # Collect tasks to compare correctly
    tasks = []
    for idx, obj in enumerate(benchmark):
        sample_num = idx + 1
        gen_path = str(Path(wav_dir) / f"sample_{sample_num:04d}_generated.wav")
        if not os.path.exists(gen_path):
            # Try 2rd variant in case it's in the subfolder or named differently
            if not os.path.exists(gen_path.replace("wav", "mp3")):
                continue
            gen_path = gen_path.replace("wav", "mp3")

        # Find a reference sample for this speaker (different from the current one if possible)
        spk_indices = spk_to_indices[obj["speaker_id"]]
        if len(spk_indices) > 1:
            # Pick the previous or next one in the speaker's list
            spk_pos = spk_indices.index(idx)
            ref_idx = spk_indices[spk_pos - 1]
        else:
            # Fallback to same sample if it's the only one (self-similarity)
            ref_idx = idx

        tasks.append((idx, ref_idx, gen_path))

    for idx, ref_idx, gen_path in tqdm(tasks):
        obj = benchmark[idx]
        ref_obj = benchmark[ref_idx]
        ref_audio = ref_obj["audio"]
        gen_dialect = obj["dialect"]

        spksim_obj = {
            "ref_path": ref_audio["path"],
            "gen_path": gen_path,
        }

        if not single:
            # Decode manually using torchaudio and io.BytesIO to avoid broken torchcodec
            wav1, sr1 = torchaudio.load(io.BytesIO(ref_audio["bytes"]))
        else:  # single prompt case
            wav1, sr1 = torchaudio.load(get_single_prompt(gen_dialect)[0])

        try:
            wav2, sr2 = torchaudio.load(gen_path)
        except RuntimeError:
            continue

        wav1 = wav1.to(device)
        wav2 = wav2.to(device)

        if sr1 != 16000:
            resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000).to(device)
            wav1 = resample1(wav1)
        if sr2 != 16000:
            resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000).to(device)
            wav2 = resample2(wav2)

        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()

        spksim_obj["spksim"] = sim
        spksim_objs.append(spksim_obj)

    return spksim_objs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav-dir", type=str, required=True)
    parser.add_argument("-d", "--dialect", type=str, required=True, help="MSA | SAU | UAE | ALG | IRQ | EGY | MAR")
    parser.add_argument("-c", "--ckpt", type=str, default="../checkpoints/UniSpeech/wavlm_large_finetune.pth")
    parser.add_argument("-s", "--single", action="store_true", help="To compare with 11labs, single reference prompt")
    args = parser.parse_args()

    spksim_objs = calculate_spksim(args.wav_dir, args.dialect, args.ckpt, args.single)

    spksim_result_path = Path(args.wav_dir) / "_spksim_results.jsonl"
    sim_scores = []
    with open(spksim_result_path, "w", encoding="utf-8") as f:
        for spksim_obj in spksim_objs:
            sim_scores.append(spksim_obj["spksim"])
            f.write(json.dumps(spksim_obj, ensure_ascii=False) + "\n")
        f.write(f"\nSPK-SIM: {np.mean(sim_scores)}\n")

    print(f"SPK-SIM: {np.mean(sim_scores)}")
    print(f"SPK-SIM results saved to {spksim_result_path}")
