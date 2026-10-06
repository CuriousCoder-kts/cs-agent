"""cs_agent.config — 配置管理（单一事实来源）

设计要点：
1. .env 显式按"仓库根目录"定位，不依赖当前工作目录——从任何位置运行都能读到；
2. frozen dataclass：配置加载后不可变，避免运行中被意外篡改；
3. require() 在昂贵调用前做廉价校验（把看不懂的报错翻译成人话）。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 仓库根目录（cs_agent/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_knowledge_dir() -> Path:
    """知识库目录：.env 的 KNOWLEDGE_DIR 优先（相对仓库根解析），否则默认 knowledge/。"""
    raw = os.getenv("KNOWLEDGE_DIR", "").strip()
    if not raw:
        return PROJECT_ROOT / "knowledge"
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    intent_confidence_threshold: float = 0.6
    knowledge_dir: Path = PROJECT_ROOT / "knowledge"

    @classmethod
    def load(cls) -> "Config":
        """从仓库根目录的 .env 加载配置。"""
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", ""),
            intent_confidence_threshold=float(
                os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.6")
            ),
            knowledge_dir=_resolve_knowledge_dir(),
        )

    def require(self) -> None:
        """配置自检：缺失或未填时，用人话报错并退出（不甩堆栈）。"""
        missing = [
            name for name, val in [
                ("LLM_BASE_URL", self.base_url),
                ("LLM_API_KEY", self.api_key),
                ("LLM_MODEL", self.model),
            ]
            if not val or val == "your_api_key_here"
        ]
        if missing:
            raise SystemExit(
                f"\n[配置错误] 缺失或未修改：{', '.join(missing)}\n"
                f"请检查 cs-agent/.env（没有就 cp .env.example .env，并填入真实 Key）。\n"
            )
