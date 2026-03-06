

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

    num_epochs: int = 20
    early_stopping_patience: int = 10 #should increase it 
    batch_size: int = 6
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 300
    save_steps: int = 2000
    eval_steps: int = 50
    logging_steps: int = 10

    sample_rate: int = 24000
    
    # LoRA Config
    lora_rank: int = 256
    lora_alpha: int = 512
    # Comma-separated string for simplicity in CLI arguments, or list in code
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

@dataclass
class ASRConfig:
    data_dir: str = "/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_train"
    test_data_dir: str = "/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_test"
    diffro_model_path: str = "./checkpoints/reward_model/token2text_reward.pt"
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

    num_epochs: int = 20
    early_stopping_patience: int = 5 #should increase it 
    batch_size: int = 6 # Reduced for DiffRO stability
    sampling_temperature: float = 1.2
    gradient_accumulation_steps: int = 8 # Reduced for DiffRO stability
    learning_rate: float = 1e-5
    warmup_steps: int = 300
    save_steps: int = 650
    eval_steps: int = 100
    logging_steps: int = 10
    
    sample_rate: int = 24000
    
    # LoRA Config
    lora_rank: int = 256
    lora_alpha: int = 512
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
