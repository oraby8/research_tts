import os

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

import argparse
import json
import re
import regex
from pathlib import Path

import editdistance
import torch
from datasets import load_dataset, Audio
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from tqdm import tqdm


def normalize_arabic_text(text):
    text = regex.sub(r"[\p{p}\p{s}]", "", text)

    # Remove diacritics
    diacritics = r"[\u064B-\u0652]"  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, "", text)

    # Normalize Hamzas and Maddas
    text = re.sub("پ", "ب", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub(r"[آ]", "ا", text)
    text = re.sub(r"[أإ]", "ا", text)
    text = re.sub(r"[ؤ]", "و", text)
    text = re.sub(r"[ئ]", "ي", text)
    text = re.sub(r"[ء]", "", text)
    text = re.sub(r"[ة]", "ه", text)

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    # Remove tatweel (kashida, u+0640)
    text = re.sub(r"\u0640", "", text)

    # Remove hmm-uhm-like words
    text = re.sub(r"اا+", "", text)

    # Normalize multiple whitespace characters into a single space
    text = re.sub(r"\s\s+", " ", text)

    return text.strip()


def word_error_rate(hypotheses, references, use_cer=False):
    scores = 0
    words = 0
    assert len(hypotheses) == len(references)
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")
    return wer


device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

lang_map = {
    "MSA": "arb_Arab",
    "SAU-Najdi": "ars_Arab",
    "SAU-Hijazi": "acw_Arab",
    "SAU-Gulf": "afb_Arab",
    "UAE": "afb_Arab",
    "ALG": "arq_Arab",
    "IRQ": "ayp_Arab",
    "EGY": "arz_Arab",
    "MAR": "ary_Arab",
    # "OMN": "acx_Arab",  # not yet supported by omniASR
    "TUN": "aeb_Arab",
    "LEV": "apc_Arab",
    "SDN": "apd_Arab",
    "LBY": "ayl_Arab",
}


def calculate_wer(pipeline, wav_dir, dialect, batch_size):
    benchmark = load_dataset("SWivid/Habibi", dialect, split="test")
    benchmark = benchmark.cast_column("audio", Audio(decode=False))

    audio_path_batch = []
    gt_text_batch = []
    lang_batch = []

    wer_objs = []
    for idx, b in enumerate(tqdm(benchmark)):
        gt_text = b["text"]

        sample_num = idx + 1
        audio_path = f"{wav_dir}/sample_{sample_num:04d}_generated.wav"
        if not os.path.exists(audio_path):
            continue

        audio_path_batch.append(audio_path)
        gt_text_batch.append(gt_text)
        lang_batch.append(lang_map.get(b["dialect"], "arb_Arab"))
        if len(audio_path_batch) < batch_size:
            continue

        transcriptions = pipeline.transcribe(
            audio_path_batch,
            lang=lang_batch,
            batch_size=batch_size,
        )

        for i in range(len(audio_path_batch)):
            gt_text_norm = normalize_arabic_text(gt_text_batch[i])
            pr_text_norm = normalize_arabic_text(transcriptions[i])
            wer_obj = {
                "audio_path": audio_path_batch[i],
                "gt_text_norm": gt_text_norm,
                "pr_text_norm": pr_text_norm,
            }
            wer = word_error_rate([pr_text_norm], [gt_text_norm])
            wer_obj["wer"] = wer
            wer_objs.append(wer_obj)

        audio_path_batch = []
        gt_text_batch = []
        lang_batch = []

    # Process remaining files in the last batch
    if len(audio_path_batch) > 0:
        transcriptions = pipeline.transcribe(
            audio_path_batch,
            lang=lang_batch,
            batch_size=len(audio_path_batch),
        )
        for i in range(len(audio_path_batch)):
            gt_text_norm = normalize_arabic_text(gt_text_batch[i])
            pr_text_norm = normalize_arabic_text(transcriptions[i])
            wer_obj = {
                "audio_path": audio_path_batch[i],
                "gt_text_norm": gt_text_norm,
                "pr_text_norm": pr_text_norm,
            }
            wer = word_error_rate([pr_text_norm], [gt_text_norm])
            wer_obj["wer"] = wer
            wer_objs.append(wer_obj)

    return wer_objs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav-dir", type=str, required=True)
    parser.add_argument(
        "-d",
        "--dialect",
        type=str,
        required=True,
        help="MSA | SAU | UAE | ALG | IRQ | EGY | MAR",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    args = parser.parse_args()

    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")  # "omniASR_LLM_7B_v2"
    wer_objs = calculate_wer(pipeline, args.wav_dir, args.dialect, args.batch_size)

    wer_result_path = Path(args.wav_dir) / "_wer_o_results.jsonl"
    pr_text_norms = []
    gt_text_norms = []
    with open(wer_result_path, "w", encoding="utf-8") as f:
        for wer_obj in wer_objs:
            pr_text_norms.append(wer_obj["pr_text_norm"])
            gt_text_norms.append(wer_obj["gt_text_norm"])
            f.write(json.dumps(wer_obj, ensure_ascii=False) + "\n")
        f.write(f"\nGlobal WER-O: {word_error_rate(pr_text_norms, gt_text_norms)}\n")

    print(f"Global WER-O: {word_error_rate(pr_text_norms, gt_text_norms)}")
    print(f"Single WER-O results saved to {wer_result_path}")
