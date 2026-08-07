# 표준 라이브러리: 난수 시드 고정과 경로 생성을 위해 사용합니다.
import os
import random

# 타입 힌트: 함수 입력/출력 타입을 명시해 가독성과 유지보수성을 높입니다.
from typing import Any

# PyTorch: 텐서 연산, dtype 선택, CUDA 사용 가능 여부 확인에 사용합니다.
import torch
# Hugging Face datasets: 데이터셋 로드/전처리(map, filter, split)에 사용합니다.
from datasets import Dataset, load_dataset
# PEFT: QLoRA 학습 준비와 LoRA 어댑터 장착에 사용합니다.
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Transformers: 모델/토크나이저 로드, 양자화 설정, 시드 고정에 사용합니다.
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
# TRL: SFTConfig/SFTTrainer 기반의 지도학습 파인튜닝에 사용합니다.
from trl import SFTConfig, SFTTrainer

# ----------------------------------------------------------------------------------
# 1. 전역 설정
# ----------------------------------------------------------------------------------
# 학습할 베이스 모델 ID입니다.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# 학습 데이터셋 ID입니다.
DATASET_ID = "gathnex/Gath_baize"
# 사용할 데이터셋 split입니다.
DATASET_SPLIT = "train"
# 학습용 샘플 개수 제한입니다. 0 이하로 설정하면 전체 데이터를 사용합니다.
MAX_SAMPLES = 200
# 검증 데이터 비율입니다.
TEST_SIZE = 0.1
# 재현성을 위한 고정 시드입니다.
RANDOM_SEED = 42
# 체크포인트 저장 경로입니다.
OUTPUT_DIR = "./llama3-practice-result"
# TensorBoard 로그 저장 경로입니다.
LOG_DIR = "./logs"
# 최종 LoRA 어댑터 저장 경로입니다.
ADAPTER_DIR = "./my_first_adapter"


# ----------------------------------------------------------------------------------
# 2. 공통 유틸리티
# ----------------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """
    학습 재현성을 높이기 위해 주요 난수 발생기를 같은 시드로 고정합니다.
    """
    # Python 기본 난수 생성기 시드를 고정합니다.
    random.seed(seed)
    # PyTorch CPU 난수 생성기 시드를 고정합니다.
    torch.manual_seed(seed)
    # CUDA가 있으면 모든 GPU 난수 생성기 시드도 고정합니다.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Transformers 유틸리티 시드도 함께 고정합니다.
    set_seed(seed)


def get_compute_dtype() -> torch.dtype:
    """
    GPU 환경에 맞는 계산 dtype을 선택합니다.
    """
    # bf16을 지원하면 bf16을 사용해 안정성과 속도를 함께 노립니다.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # bf16 미지원 CUDA 환경에서는 fp16을 사용합니다.
    if torch.cuda.is_available():
        return torch.float16
    # CUDA가 없는 환경에서는 float32를 반환합니다.
    return torch.float32


def build_tokenizer(model_id: str):
    """
    학습에 필요한 토크나이저를 구성합니다.
    """
    # 모델 ID로 토크나이저를 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id,  # 불러올 토크나이저의 허브 ID 또는 로컬 경로입니다.
    )
    # pad_token이 없으면 eos_token을 pad_token으로 맞춰 배치 패딩 경고를 방지합니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Causal LM 학습에서는 일반적으로 우측 패딩을 사용합니다.
    tokenizer.padding_side = "right"
    # 설정이 완료된 토크나이저를 반환합니다.
    return tokenizer


def convert_to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """
    다양한 대화 데이터 형식을 TRL/Transformers 표준 messages 형식으로 통일합니다.
    """
    # 최종 변환 메시지를 저장할 리스트입니다.
    messages: list[dict[str, str]] = []

    # 이미 표준 형식(messages)이면 role/content를 검증하며 그대로 정리합니다.
    if isinstance(example.get("messages"), list):
        for turn in example["messages"]:
            # 각 턴이 딕셔너리 형태인지 먼저 확인합니다.
            if not isinstance(turn, dict):
                continue
            # role 필드를 읽습니다.
            role = turn.get("role")
            # content 필드를 읽습니다.
            content = turn.get("content")
            # role이 허용된 값이 아니면 제외합니다.
            if role not in {"system", "user", "assistant"}:
                continue
            # content가 문자열이 아니면 제외합니다.
            if not isinstance(content, str):
                continue
            # 앞뒤 공백을 제거합니다.
            content = content.strip()
            # 빈 문자열이면 제외합니다.
            if not content:
                continue
            # 검증이 끝난 메시지를 누적합니다.
            messages.append({"role": role, "content": content})
        # messages 경로 처리를 마치면 결과를 반환합니다.
        return messages

    # gathnex/Gath_baize 같은 conversations 형식도 표준 형식으로 변환합니다.
    if isinstance(example.get("conversations"), list):
        for turn in example["conversations"]:
            # 각 턴이 딕셔너리 형태인지 확인합니다.
            if not isinstance(turn, dict):
                continue
            # 원본 화자 정보를 읽습니다.
            source_role = turn.get("from")
            # 원본 텍스트를 읽습니다.
            content = turn.get("value")
            # content가 문자열이 아니면 제외합니다.
            if not isinstance(content, str):
                continue
            # 앞뒤 공백을 제거합니다.
            content = content.strip()
            # 빈 문자열이면 제외합니다.
            if not content:
                continue
            # 원본 화자명을 Transformers 표준 role로 매핑합니다.
            if source_role == "human":
                role = "user"
            elif source_role in {"gpt", "assistant"}:
                role = "assistant"
            elif source_role == "system":
                role = "system"
            else:
                # 정의되지 않은 화자 값은 제외합니다.
                continue
            # 변환된 메시지를 누적합니다.
            messages.append({"role": role, "content": content})
        # conversations 경로 처리를 마치면 결과를 반환합니다.
        return messages

    # gathnex/Gath_baize의 chat_sample처럼 [INST]/[/INST] 마커가 섞인 단일 문자열도 변환합니다.
    chat_sample = example.get("chat_sample")
    if isinstance(chat_sample, str) and "[INST]" in chat_sample:
        # 첫 [INST] 앞의 안내 문구는 대화가 아니므로 버리고, [INST] 단위로 턴을 나눕니다.
        for block in chat_sample[chat_sample.index("[INST]"):].split("[INST]"):
            # [/INST]가 없는 조각은 완결된 한 턴이 아니므로 제외합니다.
            if "[/INST]" not in block:
                continue
            # 한 턴을 사용자 발화와 어시스턴트 발화로 분리합니다.
            user_content, _, assistant_content = block.partition("[/INST]")
            # 사용자 발화의 앞뒤 공백을 제거합니다.
            user_content = user_content.strip()
            # 어시스턴트 발화의 앞뒤 공백을 제거합니다.
            assistant_content = assistant_content.strip()
            # 한쪽이라도 비어 있으면 학습 쌍으로 쓸 수 없어 제외합니다.
            if not user_content or not assistant_content:
                continue
            # 사용자 메시지를 누적합니다.
            messages.append({"role": "user", "content": user_content})
            # 어시스턴트 메시지를 누적합니다.
            messages.append({"role": "assistant", "content": assistant_content})

    # 변환 결과를 반환합니다.
    return messages


def make_format_chat_fn(tokenizer):
    """
    tokenizer.apply_chat_template 기반 포맷팅 함수를 생성합니다.
    """

    def format_chat(example: dict[str, Any]) -> dict[str, str]:
        # 샘플을 표준 messages 형식으로 변환합니다.
        messages = convert_to_messages(example)
        # user 메시지 존재 여부를 확인합니다.
        has_user = any(message["role"] == "user" for message in messages)
        # assistant 메시지 존재 여부를 확인합니다.
        has_assistant = any(message["role"] == "assistant" for message in messages)
        # 양쪽 화자가 모두 있어야 SFT 학습 샘플로 유효합니다.
        if not (has_user and has_assistant):
            return {"text": ""}
        # 공식 chat template를 적용해 학습 텍스트를 생성합니다.
        text = tokenizer.apply_chat_template(
            conversation=messages,             # role/content 구조의 전체 대화 메시지 리스트입니다.
            tokenize=False,                    # 문자열을 바로 반환하고 토큰 ID 텐서는 만들지 않습니다.
            add_generation_prompt=False,       # 학습용 텍스트이므로 마지막 assistant 시작 토큰은 추가하지 않습니다.
        )
        # SFTTrainer가 text 컬럼을 다시 토크나이즈하며 BOS를 붙이므로, 템플릿이 넣은 BOS를 제거해 중복을 막습니다.
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text.removeprefix(tokenizer.bos_token)
        # 텍스트 필드를 반환합니다.
        return {"text": text}

    # 내부 포맷팅 함수를 반환합니다.
    return format_chat


def prepare_datasets(tokenizer) -> tuple[Dataset, Dataset]:
    """
    데이터셋 로드, 샘플링, 포맷팅, 유효성 필터링, train/eval 분할을 수행합니다.
    """
    # 원본 데이터셋을 로드합니다.
    raw_dataset = load_dataset(
        path=DATASET_ID,           # 불러올 데이터셋의 허브 ID 또는 로컬 스크립트 경로입니다.
        split=DATASET_SPLIT,       # 사용할 split 이름입니다. 예: train, validation, test
    )
    # 샘플 수 제한이 설정되어 있으면 먼저 셔플 후 일부만 사용합니다.
    if MAX_SAMPLES > 0:
        # 앞부분 편향을 줄이기 위해 시드 고정 셔플을 수행합니다.
        raw_dataset = raw_dataset.shuffle(
            seed=RANDOM_SEED,      # 셔플 순서를 고정해 실행마다 같은 샘플 순서를 만듭니다.
        )
        # 실제 사용할 샘플 수를 계산합니다.
        sample_count = min(MAX_SAMPLES, len(raw_dataset))
        # 계산된 개수만큼 잘라서 사용합니다.
        raw_dataset = raw_dataset.select(
            indices=range(sample_count),   # 유지할 샘플 인덱스 집합입니다.
        )

    # chat template 적용 함수를 생성합니다.
    format_chat = make_format_chat_fn(tokenizer)
    # 전체 샘플에 템플릿을 적용하고, 원본 컬럼은 제거해 관리 포인트를 줄입니다.
    processed_dataset = raw_dataset.map(
        function=format_chat,                    # 각 샘플을 {"text": "..."}로 변환하는 함수입니다.
        remove_columns=raw_dataset.column_names, # 원본 컬럼을 제거하고 text 컬럼만 남깁니다.
        desc="chat template 적용",               # 진행 바(progress bar)에 표시할 작업 설명입니다.
    )
    # 빈 텍스트 샘플을 제거해 학습 안정성을 높입니다.
    processed_dataset = processed_dataset.filter(
        function=lambda item: len(item["text"]) > 0,   # text가 빈 문자열이 아닌 샘플만 유지합니다.
        desc="유효 샘플 필터링",                        # 진행 바(progress bar)에 표시할 작업 설명입니다.
    )
    # 학습/검증으로 분할합니다.
    split_dataset = processed_dataset.train_test_split(
        test_size=TEST_SIZE,     # 검증(test) 데이터로 분리할 비율입니다.
        seed=RANDOM_SEED,        # 분할 난수를 고정해 같은 train/eval 분할을 재현합니다.
    )
    # train split을 꺼냅니다.
    train_dataset = split_dataset["train"]
    # test split을 eval 데이터셋으로 사용합니다.
    eval_dataset = split_dataset["test"]
    # 준비된 두 데이터셋을 반환합니다.
    return train_dataset, eval_dataset


def build_model(model_id: str):
    """
    QLoRA 학습용 베이스 모델을 로드하고 LoRA 어댑터를 장착합니다.
    """
    # QLoRA 4bit 학습은 CUDA 환경을 전제로 합니다.
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 학습은 CUDA GPU 환경에서 실행해야 합니다.")

    # 실행 환경에 맞는 계산 dtype을 선택합니다.
    compute_dtype = get_compute_dtype()
    # BitsAndBytes 4bit 양자화 설정을 구성합니다.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 베이스 모델 가중치를 4bit 형식으로 로드합니다.
        bnb_4bit_use_double_quant=True,       # 양자화 상수를 한 번 더 양자화해 메모리를 절약합니다.
        bnb_4bit_quant_type="nf4",            # 4bit 양자화 타입으로 NF4를 사용합니다.
        bnb_4bit_compute_dtype=compute_dtype, # 실제 연산에 사용할 dtype(fp16/bf16)을 지정합니다.
    )
    # 양자화된 베이스 모델을 로드합니다.
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,  # 불러올 베이스 Causal LM 모델 ID입니다.
        quantization_config=bnb_config,          # 위에서 만든 4bit 양자화 설정을 적용합니다.
        device_map="auto",                       # 가용 GPU/CPU에 레이어를 자동 배치합니다.
        torch_dtype=compute_dtype,               # 모델 연산에 사용할 기본 dtype을 지정합니다.
    )
    # k-bit 학습 준비를 수행합니다.
    model = prepare_model_for_kbit_training(
        model,                               # 양자화가 적용된 베이스 모델 객체입니다.
        use_gradient_checkpointing=True,     # 체크포인팅을 켜서 VRAM 사용량을 줄입니다.
    )
    # 학습 중 캐시 사용은 메모리 증가를 유발하므로 비활성화합니다.
    model.config.use_cache = False

    # LoRA 설정을 정의합니다.
    lora_config = LoraConfig(
        r=16,                                                      # LoRA 저랭크 행렬의 rank입니다.
        lora_alpha=32,                                             # LoRA 스케일링 계수입니다.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 어댑터를 삽입할 대상 모듈 이름 목록입니다.
        lora_dropout=0.05,                                         # LoRA 경로에 적용할 드롭아웃 비율입니다.
        bias="none",                                               # bias 파라미터는 학습하지 않도록 설정합니다.
        task_type="CAUSAL_LM",                                     # 태스크 타입을 Causal Language Modeling으로 지정합니다.
    )
    # 베이스 모델에 LoRA 어댑터를 장착합니다.
    model = get_peft_model(
        model=model,                 # LoRA를 장착할 베이스 모델입니다.
        peft_config=lora_config,     # 위에서 정의한 LoRA 설정 객체입니다.
    )
    # 학습될 파라미터 비율을 출력해 설정이 맞는지 확인합니다.
    model.print_trainable_parameters()
    # LoRA가 장착된 모델을 반환합니다.
    return model


def build_training_args() -> SFTConfig:
    """
    SFT 학습 하이퍼파라미터를 구성합니다.
    """
    # 환경에 맞춰 mixed precision 옵션을 계산합니다.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # bf16 미지원 CUDA에서는 fp16을 사용합니다.
    use_fp16 = torch.cuda.is_available() and not use_bf16
    # SFTConfig를 생성합니다.
    return SFTConfig(
        output_dir=OUTPUT_DIR,                         # 체크포인트와 상태 파일을 저장할 경로입니다.
        per_device_train_batch_size=2,                # 디바이스(GPU) 1개당 학습 배치 크기입니다.
        per_device_eval_batch_size=2,                 # 디바이스(GPU) 1개당 평가 배치 크기입니다.
        gradient_accumulation_steps=4,                # 몇 step의 그래디언트를 누적 후 업데이트할지 지정합니다.
        learning_rate=2e-4,                           # 옵티마이저의 초기 학습률입니다.
        num_train_epochs=1,                           # 전체 데이터셋 반복 횟수(epoch)입니다.
        logging_steps=5,                              # loss/log를 몇 step마다 기록할지 지정합니다.
        bf16=use_bf16,                                # bf16 mixed precision 사용 여부입니다.
        fp16=use_fp16,                                # fp16 mixed precision 사용 여부입니다.
        optim="paged_adamw_8bit",                     # 사용할 옵티마이저 구현체 이름입니다.
        warmup_ratio=0.03,                            # 전체 step 대비 warmup 비율입니다.
        lr_scheduler_type="cosine",                   # 학습률 스케줄러 유형입니다.
        max_length=512,                               # 학습 시퀀스 최대 길이입니다.
        dataset_text_field="text",                    # 데이터셋에서 학습 텍스트가 들어있는 컬럼 이름입니다.
        assistant_only_loss=False,                    # Llama-3 chat template에는 {% generation %} 마커가 없어 assistant 구간만 loss를 계산할 수 없습니다.
        eval_strategy="steps",                        # 평가를 step 단위로 수행하도록 지정합니다.
        eval_steps=10,                                # 평가를 몇 step마다 실행할지 지정합니다.
        save_strategy="steps",                        # 체크포인트 저장을 step 단위로 수행합니다.
        save_steps=10,                                # 체크포인트를 몇 step마다 저장할지 지정합니다.
        save_total_limit=3,                           # 유지할 체크포인트 최대 개수입니다.
        load_best_model_at_end=True,                  # 학습 종료 후 최고 성능 체크포인트를 로드합니다.
        metric_for_best_model="eval_loss",            # 최고 모델 선택 시 사용할 평가 지표 이름입니다.
        greater_is_better=False,                      # 지표가 낮을수록 좋은지(True/False) 방향을 지정합니다.
        gradient_checkpointing=True,                  # 그래디언트 체크포인팅 활성화 여부입니다.
        seed=RANDOM_SEED,                             # 학습 관련 난수 시드입니다.
        data_seed=RANDOM_SEED,                        # 데이터 샘플링/셔플 관련 난수 시드입니다.
        report_to="tensorboard",                      # 학습 로그를 전송할 리포터 백엔드입니다.
        logging_dir=LOG_DIR,                          # TensorBoard 로그를 저장할 경로입니다.
        packing=False,                                # 여러 샘플을 한 시퀀스로 묶는 example packing 사용 여부입니다.
    )


def main() -> None:
    """
    학습 파이프라인 전체를 실행합니다.
    """
    # 재현성을 위해 시드를 먼저 고정합니다.
    set_global_seed(seed=RANDOM_SEED)   # 전체 파이프라인에서 사용할 공통 시드값입니다.
    # 출력 디렉터리를 사전에 생성합니다.
    os.makedirs(
        name=OUTPUT_DIR,        # 생성할 디렉터리 경로입니다.
        exist_ok=True,          # 이미 디렉터리가 있어도 오류 없이 계속 진행합니다.
    )
    # 로그 디렉터리를 사전에 생성합니다.
    os.makedirs(
        name=LOG_DIR,           # TensorBoard 로그 디렉터리 경로입니다.
        exist_ok=True,          # 이미 존재할 때 예외를 발생시키지 않습니다.
    )
    # 어댑터 저장 디렉터리를 사전에 생성합니다.
    os.makedirs(
        name=ADAPTER_DIR,       # 최종 LoRA 어댑터 저장 디렉터리 경로입니다.
        exist_ok=True,          # 이미 존재할 때 예외를 발생시키지 않습니다.
    )

    # 토크나이저를 준비합니다.
    tokenizer = build_tokenizer(model_id=MODEL_ID)   # 토크나이저를 만들 베이스 모델 ID입니다.
    # 학습/검증 데이터셋을 준비합니다.
    train_dataset, eval_dataset = prepare_datasets(tokenizer)
    # 데이터셋 크기를 출력합니다.
    print(f"학습 데이터: {len(train_dataset)}개, 검증 데이터: {len(eval_dataset)}개")

    # LoRA 장착 모델을 준비합니다.
    model = build_model(MODEL_ID)
    # 학습 설정을 준비합니다.
    training_args = build_training_args()
    # SFTTrainer를 생성합니다.
    trainer = SFTTrainer(
        model=model,                      # 학습 대상인 LoRA 장착 모델입니다.
        args=training_args,               # 학습 하이퍼파라미터(SFTConfig)입니다.
        train_dataset=train_dataset,      # 학습용 데이터셋입니다.
        eval_dataset=eval_dataset,        # 평가용 데이터셋입니다.
        processing_class=tokenizer,       # 텍스트 처리(토큰화/패딩)에 사용할 토크나이저입니다.
    )

    # 학습 시작 로그를 출력합니다.
    print("--- 학습을 시작합니다 ---")
    # 실제 학습을 수행합니다.
    trainer.train()

    # 학습된 LoRA 어댑터를 저장합니다.
    model.save_pretrained(
        save_directory=ADAPTER_DIR,   # LoRA 어댑터 가중치/설정을 저장할 디렉터리입니다.
    )
    # 추론 일관성을 위해 토크나이저도 함께 저장합니다.
    tokenizer.save_pretrained(
        save_directory=ADAPTER_DIR,   # 추론 시 같은 토크나이저를 쓰기 위해 저장할 경로입니다.
    )
    # 완료 메시지를 출력합니다.
    print(f"학습 완료! 어댑터가 '{ADAPTER_DIR}'에 저장되었습니다.")


# 스크립트로 직접 실행할 때만 학습을 시작합니다.
if __name__ == "__main__":
    main()
