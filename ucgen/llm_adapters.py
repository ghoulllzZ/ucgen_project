from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_env_placeholders(value: Optional[str]) -> str:
    """
    解析 config.yaml 中形如 "${OPENAI_API_KEY}" 的占位符；未配置则返回空串。
    也兼容直接写明文 key/base_url/model 的情况。
    """
    if not value:
        return ""
    def repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return _ENV_PATTERN.sub(repl, value).strip()


def extract_plantuml(text: str) -> str:
    """
    真实 LLM 输出经常夹杂说明文字，这里强制抽取：
    1) 优先 @startuml..@enduml
    2) 其次 ```...``` 代码块
    3) 否则原样返回（让解析器报错定位）
    """
    if not text:
        return text

    # 1) @startuml ... @enduml
    s = text.find("@startuml")
    e = text.rfind("@enduml")
    if s != -1 and e != -1 and e > s:
        return (text[s : e + len("@enduml")] + "\n").strip() + "\n"

    # 2) fenced code block
    m = re.search(r"```(?:plantuml)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return (m.group(1).strip() + "\n")

    return text.strip() + "\n"


def _safe_get(d: Dict[str, Any], *keys: str, default: str = "") -> str:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return str(cur) if cur is not None else default


@dataclass
class OpenAICompatLLM:
    """
    OpenAI-Compatible 适配器：可覆盖 OpenAI / DeepSeek / Qwen(OpenAI compatible mode) 等。
    - 通过 base_url 区分不同厂商
    - 通过 api_key 区分不同 key
    """
    name: str
    api_key: str
    base_url: str
    model: str

    def generate(self, prompt: str, temperature: float, seed: int) -> str:
        # 懒加载，避免 mock 跑的时候也要求安装 openai
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "缺少依赖 openai。请先执行：pip install -r requirements.txt（或 pip install openai）"
            ) from e

        if not self.api_key:
            raise RuntimeError(f"{self.name} 未配置 api_key（环境变量或 config.yaml）")
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"
        if not self.model:
            raise RuntimeError(f"{self.name} 未配置 model")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        messages = [
            {
                "role": "system",
                "content": (
                    "You must output ONLY a valid PlantUML use-case diagram. "
                    "Do not include any extra explanation outside the diagram. "
                    "The diagram must contain @startuml and @enduml."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # 不同 OpenAI-compatible 服务对 seed 参数支持不一：先尝试带 seed，再 fallback
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                seed=seed,  # 可能不被某些兼容服务支持
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )

        content = resp.choices[0].message.content or ""
        return extract_plantuml(content)


@dataclass
class GeminiLLM:
    """
    Gemini 适配器（google-genai SDK）。
    默认从环境变量 GEMINI_API_KEY/GOOGLE_API_KEY 读取；也支持 config.yaml 提供 api_key。
    """
    name: str
    api_key: str
    model: str

    def generate(self, prompt: str, temperature: float, seed: int) -> str:
        # 懒加载，避免 mock 跑的时候也要求安装 google-genai
        try:
            from google import genai
        except Exception as e:
            raise RuntimeError(
                "缺少依赖 google-genai。请先执行：pip install -r requirements.txt（或 pip install google-genai）"
            ) from e

        # google-genai：api_key 可不传（若已设置 GEMINI_API_KEY/GOOGLE_API_KEY）
        if self.api_key:
            client = genai.Client(api_key=self.api_key)
        else:
            client = genai.Client()

        if not self.model:
            self.model = "gemini-2.5-flash"

        # google-genai 的 temperature/seed 参数并非所有接口/版本都支持；先尝试传参，不行就降级
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": temperature},
            )
        except TypeError:
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

        text = getattr(resp, "text", "") or ""
        return extract_plantuml(text)


def build_llm_from_config(name: str, cfg: Dict[str, Any]) -> Any:
    """
    根据 config.yaml 的 openai/deepseek/qwen/gemini 节点构建 LLM 实例。
    cfg 格式与 config.yaml.example 对齐：
      openai:
        api_key: "${OPENAI_API_KEY}"
        base_url: "${OPENAI_BASE_URL}"
        model: "${OPENAI_MODEL}"
    """
    block = cfg.get(name, {}) if isinstance(cfg, dict) else {}
    api_key = resolve_env_placeholders(_safe_get(block, "api_key"))
    base_url = resolve_env_placeholders(_safe_get(block, "base_url"))
    model = resolve_env_placeholders(_safe_get(block, "model"))
    # 给特定后端补默认 base_url，避免用户漏填
    if name == "doubao" and not base_url:
        # 火山方舟 Ark OpenAI-compatible base_url（v3）
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
    if name == "grok" and not base_url:
        # xAI OpenAI-compatible base_url（v1）
        base_url = "https://api.x.ai/v1"

    if name in ("openai", "deepseek", "qwen", "doubao", "grok"):
        # doubao(ark) 与 grok(xAI) 均为 OpenAI-compatible 调用方式
        return OpenAICompatLLM(name=name, api_key=api_key, base_url=base_url, model=model)

    if name == "gemini":
        # Gemini 默认也可不在 config.yaml 里写 api_key，只要环境变量设置即可
        api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        model = model or "gemini-2.5-flash"
        return GeminiLLM(name="gemini", api_key=api_key, model=model)

    raise ValueError(f"Unknown llm name: {name}")
