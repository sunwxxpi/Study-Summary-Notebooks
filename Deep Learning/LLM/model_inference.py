# 타입 힌트: 입력/출력 타입을 명시해 코드 이해를 돕습니다.
from typing import Any

# PyTorch: dtype 선택, CUDA 확인, 추론 모드 제어에 사용합니다.
import torch
# PEFT: LoRA 어댑터를 베이스 모델 위에 로드하는 데 사용합니다.
from peft import PeftModel
# Transformers: 모델/토크나이저/양자화/파이프라인/시드 고정에 사용합니다.
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, set_seed

# ----------------------------------------------------------------------------------
# 1. 경로 및 기본 설정
# ----------------------------------------------------------------------------------
# 학습 때 사용한 베이스 모델 ID입니다.
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# 학습이 끝난 LoRA 어댑터 경로입니다.
ADAPTER_PATH = "./my_first_adapter"
# 추론 시 재현성을 위한 시드입니다.
RANDOM_SEED = 42
# 히스토리 폭주를 막기 위해 유지할 최대 사용자/어시스턴트 턴 수입니다.
MAX_HISTORY_TURNS = 8


# ----------------------------------------------------------------------------------
# 2. 공통 유틸리티
# ----------------------------------------------------------------------------------
def get_compute_dtype() -> torch.dtype:
    """
    GPU 환경에 맞는 계산 dtype을 선택합니다.
    """
    # bf16 지원 GPU면 bf16을 우선 사용합니다.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # bf16 미지원 CUDA 환경이면 fp16을 사용합니다.
    if torch.cuda.is_available():
        return torch.float16
    # CUDA가 없으면 float32를 사용합니다.
    return torch.float32


def build_inference_pipeline(base_model_id: str, adapter_path: str):
    """
    베이스 모델 + LoRA 어댑터 + text-generation pipeline을 준비합니다.
    """
    # QLoRA 4bit 추론은 CUDA 환경을 전제로 합니다.
    if not torch.cuda.is_available():
        raise RuntimeError("4bit 양자화 추론은 CUDA GPU 환경에서 실행해야 합니다.")

    # 환경에 맞는 dtype을 계산합니다.
    compute_dtype = get_compute_dtype()
    # 학습과 동일한 4bit 양자화 설정을 구성합니다.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 모델 가중치를 4bit로 로드합니다.
        bnb_4bit_use_double_quant=True,       # 양자화 상수를 한 번 더 양자화해 메모리를 줄입니다.
        bnb_4bit_quant_type="nf4",            # 4bit 양자화 타입으로 NF4를 사용합니다.
        bnb_4bit_compute_dtype=compute_dtype, # 실제 계산에 사용할 dtype(fp16/bf16)을 지정합니다.
    )
    # 어댑터 경로에서 토크나이저를 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=adapter_path,   # 로드할 토크나이저 경로(어댑터 폴더)입니다.
    )
    # pad_token이 비어 있으면 eos_token으로 맞춥니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only 생성은 좌측 패딩이어야 패딩 토큰이 생성 시작 위치를 밀어내지 않습니다.
    tokenizer.padding_side = "left"

    # 베이스 모델을 양자화 설정으로 로드합니다.
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_id,  # 로드할 베이스 Causal LM 모델 ID입니다.
        quantization_config=bnb_config,               # 위에서 만든 4bit 양자화 설정입니다.
        device_map="auto",                            # 가용 GPU/CPU에 레이어를 자동 배치합니다.
        torch_dtype=compute_dtype,                    # 모델 연산에 사용할 기본 dtype입니다.
    )
    # 베이스 모델 위에 학습된 LoRA 어댑터를 결합합니다.
    model = PeftModel.from_pretrained(
        model=base_model,            # LoRA를 붙일 베이스 모델 객체입니다.
        model_id=adapter_path,       # 불러올 PEFT 어댑터(LoRA) 경로입니다.
    )
    # 추론 모드로 전환해 학습 전용 동작을 비활성화합니다.
    model.eval()

    # text-generation 파이프라인을 생성합니다.
    text_pipe = pipeline(
        task="text-generation",      # 수행할 파이프라인 태스크 유형입니다.
        model=model,                 # 실제 추론에 사용할 LoRA 결합 모델입니다.
        tokenizer=tokenizer,         # 입력/출력 텍스트 변환에 사용할 토크나이저입니다.
        torch_dtype=compute_dtype,   # 파이프라인 연산에 사용할 dtype입니다.
        device_map="auto",           # 모델을 가용 디바이스에 자동 배치합니다.
    )
    # 파이프라인과 토크나이저를 반환합니다.
    return text_pipe, tokenizer


# ----------------------------------------------------------------------------------
# 3. Multi-turn 대화 관리 클래스
# ----------------------------------------------------------------------------------
class ChatSession:
    """
    멀티턴 대화 상태(messages)와 생성 호출을 관리하는 클래스입니다.
    """

    def __init__(self, text_pipe, tokenizer, system_prompt: str | None = None, max_history_turns: int = 8):
        # text-generation 파이프라인을 저장합니다.
        self.text_pipe = text_pipe
        # 채팅 템플릿 적용을 위한 토크나이저를 저장합니다.
        self.tokenizer = tokenizer
        # 히스토리 제한 턴 수를 저장합니다.
        self.max_history_turns = max_history_turns
        # 대화 메시지 목록을 초기화합니다.
        self.messages: list[dict[str, str]] = []
        # 시스템 프롬프트가 있으면 첫 메시지로 저장합니다.
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _trim_history(self) -> None:
        """
        시스템 메시지를 유지하면서 최근 N턴만 남깁니다.
        """
        # 시스템 메시지만 분리합니다.
        system_messages = [message for message in self.messages if message["role"] == "system"]
        # 사용자/어시스턴트 메시지만 분리합니다.
        non_system_messages = [message for message in self.messages if message["role"] != "system"]
        # 유지할 최대 일반 메시지 개수를 계산합니다. (한 턴은 user+assistant 2개)
        max_non_system_messages = self.max_history_turns * 2
        # 제한을 넘으면 최근 메시지만 남깁니다.
        if len(non_system_messages) > max_non_system_messages:
            non_system_messages = non_system_messages[-max_non_system_messages:]
        # 시스템 + 최근 메시지를 합쳐 최종 히스토리를 갱신합니다.
        self.messages = system_messages + non_system_messages

    def chat(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        사용자 메시지를 받아 모델 응답을 생성하고 히스토리에 반영합니다.
        """
        # 사용자 입력을 히스토리에 추가합니다.
        self.messages.append({"role": "user", "content": user_message})
        # 히스토리 길이를 제한합니다.
        self._trim_history()

        # 공식 채팅 템플릿을 적용해 모델 입력 프롬프트를 구성합니다.
        prompt = self.tokenizer.apply_chat_template(
            conversation=self.messages,   # 시스템/사용자/어시스턴트 대화 이력을 담은 메시지 리스트입니다.
            tokenize=False,               # 토큰 ID가 아닌 문자열 프롬프트를 반환합니다.
            add_generation_prompt=True,   # 마지막에 assistant 시작 신호를 붙여 응답 생성을 유도합니다.
        )

        # 기본 생성 파라미터를 구성합니다.
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,               # 새로 생성할 최대 토큰 수입니다.
            "do_sample": do_sample,                         # 확률 샘플링 사용 여부입니다.
            "eos_token_id": self.tokenizer.eos_token_id,    # 이 토큰이 나오면 생성을 종료합니다.
            "pad_token_id": self.tokenizer.pad_token_id,    # 배치 패딩에 사용할 토큰 ID입니다.
            "return_full_text": False,                      # 프롬프트를 제외하고 생성 텍스트만 반환합니다.
        }
        # 샘플링 모드일 때만 temperature/top_p를 적용합니다.
        if do_sample:
            generation_kwargs["temperature"] = temperature   # 샘플링 분포의 날카로움을 제어합니다.
            generation_kwargs["top_p"] = top_p               # 누적 확률 top-p 범위 내 토큰만 샘플링합니다.

        # 추론 모드에서 파이프라인을 실행해 응답을 생성합니다.
        with torch.inference_mode():
            outputs = self.text_pipe(prompt, **generation_kwargs)

        # 반환 형식에서 생성 텍스트만 안전하게 꺼냅니다.
        response = outputs[0]["generated_text"].strip()
        # 생성된 응답을 히스토리에 저장합니다.
        self.messages.append({"role": "assistant", "content": response})
        # 모델 응답을 반환합니다.
        return response

    def reset(self) -> None:
        """
        시스템 메시지를 제외한 대화 히스토리를 초기화합니다.
        """
        # 시스템 메시지만 남기고 모두 제거합니다.
        self.messages = [message for message in self.messages if message["role"] == "system"]

    def get_history(self) -> list[dict[str, str]]:
        """
        현재 히스토리의 복사본을 반환합니다.
        """
        # 외부에서 원본을 직접 수정하지 못하도록 복사본을 반환합니다.
        return self.messages.copy()


# ----------------------------------------------------------------------------------
# 4. 실행 예시
# ----------------------------------------------------------------------------------
def main() -> None:
    """
    로컬 멀티턴 대화 예시를 실행합니다.
    """
    # 시드를 고정해 샘플링 변동성을 줄입니다.
    set_seed(
        seed=RANDOM_SEED,      # 샘플링 기반 생성의 재현성을 높이기 위한 시드값입니다.
    )
    # 파이프라인과 토크나이저를 로드합니다.
    text_pipe, tokenizer = build_inference_pipeline(
        base_model_id=BASE_MODEL_ID,   # 어댑터가 학습된 원본 베이스 모델 ID입니다.
        adapter_path=ADAPTER_PATH,     # 로컬에 저장된 LoRA 어댑터 경로입니다.
    )
    # 대화 세션을 생성합니다.
    session = ChatSession(
        text_pipe=text_pipe,                    # 실제 생성 호출을 담당하는 text-generation 파이프라인입니다.
        tokenizer=tokenizer,                    # 채팅 템플릿 적용과 토큰 처리를 담당하는 토크나이저입니다.
        max_history_turns=MAX_HISTORY_TURNS,    # 유지할 최근 대화 턴 수 제한값입니다.
    )

    # 콘솔 안내 문구를 출력합니다.
    print("=== Multi-turn 대화 테스트 ===\n")

    # 첫 번째 질문을 보냅니다.
    q1 = "파이썬이 뭐야?"
    print(f"[Turn 1] 사용자: {q1}")
    a1 = session.chat(q1)
    print(f"[Turn 1] 모델: {a1}\n")
    print("-" * 50)

    # 두 번째 질문으로 맥락 유지 여부를 확인합니다.
    q2 = "그걸로 뭘 만들 수 있어?"
    print(f"\n[Turn 2] 사용자: {q2}")
    a2 = session.chat(q2)
    print(f"[Turn 2] 모델: {a2}\n")
    print("-" * 50)

    # 세 번째 질문을 보냅니다.
    q3 = "초보자가 배우기 어려워?"
    print(f"\n[Turn 3] 사용자: {q3}")
    a3 = session.chat(q3)
    print(f"[Turn 3] 모델: {a3}\n")
    print("-" * 50)

    # 네 번째 질문을 보냅니다.
    q4 = "파이썬으로 1부터 100까지 더하는 가장 효율적인 방법은?"
    print(f"\n[Turn 4] 사용자: {q4}")
    a4 = session.chat(q4)
    print(f"[Turn 4] 모델: {a4}")


# 스크립트로 직접 실행할 때만 예시를 수행합니다.
if __name__ == "__main__":
    main()
