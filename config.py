

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FinetuneConfig:
    data_dir: str = "/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_train"
    test_data_dir: str = "/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_test"
    output_dir: str = "./output_trainig_partial"
    report_project_name: str = "chatterbox_new"

    # ✅ RESUME SUPPORT - Set this to resume training!
    resume_from_checkpoint: Optional[str] = None

    device: str = "cuda"
    language_id: str = "ar"

    freeze_text_encoder: bool = True
    text_encoder_lr: float = 2e-5
    freeze_early_acoustic: bool = True
    unfreeze_prosody: bool = True
    unfreeze_late_acoustic: bool = True

    num_epochs: int = 10
    early_stopping_patience: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 300
    save_steps: int = 2000
    eval_steps: int = 50
    logging_steps: int = 10

    sample_rate: int = 24000

@dataclass
class ASRConfig:
    data_dir: str = "/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_train"
    test_data_dir: str = "/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_test"
    output_dir: str = "./output_trainig_partial"
    report_project_name: str = "chatterbox_new"

    # ✅ RESUME SUPPORT - Set this to resume training!
    resume_from_checkpoint: Optional[str] = None

    device: str = "cuda"
    language_id: str = "ar"

    freeze_text_encoder: bool = True
    text_encoder_lr: float = 2e-5
    freeze_early_acoustic: bool = True
    unfreeze_prosody: bool = True
    unfreeze_late_acoustic: bool = True

    num_epochs: int = 10
    early_stopping_patience: int = 5
    batch_size: int = 4
    sampling_temperature: float = 1.2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 300
    save_steps: int = 650
    eval_steps: int = 100
    logging_steps: int = 10

    sample_rate: int = 24000
