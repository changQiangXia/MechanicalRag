"""LLM 加载模块：支持 Qwen2（中文）与 Flan-T5，统一返回 LangChain 兼容的 LLM。"""

from __future__ import annotations

import torch
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from model_provider import resolve_model_path


# 默认中文模型
DEFAULT_CN_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_EN_MODEL = "google/flan-t5-base"


class Qwen2PipelineWrapper:
    """Qwen2 聊天模型包装器：将 RAG 提示转为 chat 格式后生成。"""

    def __init__(self, model_name: str, max_new_tokens: int = 256, **kwargs):
        model_path = resolve_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_kw = {"trust_remote_code": True, **kwargs}
        if self.device == "cuda":
            load_kw["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = 1.0
            self.model.generation_config.top_p = 1.0
            self.model.generation_config.top_k = 50
        self.max_new_tokens = max_new_tokens
        self.task = "text-generation"

    def __call__(self, prompts, **kwargs):
        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)

        outputs = []
        for prompt in prompts:
            messages = [
                {
                    "role": "system",
                    "content": "你是机械工程与具身智能领域的专家助手。请根据上下文准确、简洁地回答问题。若上下文无相关信息，请明确说明。",
                },
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            outputs.append({"generated_text": gen.strip()})

        return outputs if not single else outputs


def get_llm(
    model_name: str | None = None,
    use_chinese: bool = True,
    max_new_tokens: int = 256,
):
    """
    根据模型名返回 LangChain 兼容的 LLM。
    - Qwen2：中文能力强，因果语言模型
    - Flan-T5：Seq2Seq，英文较好
    """
    if model_name is None:
        model_name = DEFAULT_CN_MODEL if use_chinese else DEFAULT_EN_MODEL

    if "qwen" in model_name.lower() or "Qwen" in model_name:
        pipe = Qwen2PipelineWrapper(model_name, max_new_tokens=max_new_tokens)
        return HuggingFacePipeline(pipeline=pipe)
    else:
        model_path = resolve_model_path(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            repetition_penalty=1.2,
        )
        try:
            _ = pipe.model  # 兼容旧 pipeline
        except AttributeError:
            pass

        class Text2TextWrapper:
            def __init__(self, p):
                self._pipe = p
                self.task = p.task

            def __call__(self, prompts, **kw):
                kw.pop("return_full_text", None)
                return self._pipe(prompts, **kw)

        return HuggingFacePipeline(pipeline=Text2TextWrapper(pipe))
