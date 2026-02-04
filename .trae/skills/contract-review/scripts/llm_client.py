"""
合同审核 LLM 调用客户端
支持 Qwen、Qwen-VL (OCR)、DeepSeek API

版本: 1.1.0
更新: 2025-02-04
"""

import os
import re
import json
import base64
import logging
import time
from pathlib import Path
from typing import Optional, Literal, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 可选依赖
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    logger.warning("httpx 未安装，请运行: pip install httpx")


@dataclass
class LLMConfig:
    """LLM 配置"""
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class LLMResponse:
    """LLM 响应结果"""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)


# 分析任务的提示词模板
ANALYSIS_PROMPTS = {
    "full_review": {
        "system": """你是一位专业的合同审核专家，具有丰富的法律知识和合同审查经验。
请对提供的合同进行全面审核，包括：
1. 识别合同类型和基本信息
2. 分析各条款的风险点
3. 检查合规性问题
4. 提供修改建议

请以结构化的方式输出审核结果。""",
        "user": "请对以下合同进行全面审核：",
    },
    "risk_analysis": {
        "system": """你是一位专业的合同风险分析师。
请专注于识别合同中的风险条款，包括：
- 高风险：可能导致重大损失的条款
- 中风险：需要关注的潜在问题
- 低风险：建议优化的条款

对每个风险点，请说明风险原因和修改建议。""",
        "user": "请分析以下合同中的风险点：",
    },
    "clause_extract": {
        "system": """你是一位合同信息提取专家。
请从合同中提取以下关键信息，严格以 JSON 格式输出（不要使用 markdown 代码块包裹）：
{
    "contract_type": "合同类型",
    "parties": {"party_a": "甲方", "party_b": "乙方"},
    "amount": "合同金额",
    "duration": "合同期限",
    "key_obligations": ["主要义务1", "主要义务2"],
    "termination_conditions": ["解除条件1"],
    "dispute_resolution": "争议解决方式"
}""",
        "user": "请从以下合同中提取关键信息：",
    },
    "compliance_check": {
        "system": """你是一位法律合规专家。
请检查合同是否符合相关法律法规要求，包括：
1. 必备条款是否齐全
2. 是否存在违反强制性法规的条款
3. 格式条款是否履行了提示说明义务
4. 特殊事项是否需要审批备案
5. 个人信息处理是否符合《个人信息保护法》""",
        "user": "请检查以下合同的合规性：",
    },
    "ocr": {
        "system": """你是一位专业的文档识别专家。
请识别图片中的合同文本内容，要求：
1. 完整提取所有文字，保持原有格式和段落结构
2. 表格内容用 | 分隔符表示
3. 如有印章、签名等，用 [印章]、[签名] 标注位置
4. 模糊或无法识别的文字用 [?] 标注""",
        "user": "请识别以下合同图片中的文字内容：",
    },
}


class BaseLLMClient(ABC):
    """LLM 客户端基类"""

    BASE_URL: str = ""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        config: Optional[LLMConfig] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.config = config or LLMConfig()
        self._validate_api_key()

    @abstractmethod
    def _get_api_key_env_vars(self) -> list:
        """返回 API Key 环境变量名列表"""
        pass

    def _validate_api_key(self):
        """验证 API Key"""
        if not self.api_key:
            for env_var in self._get_api_key_env_vars():
                self.api_key = os.getenv(env_var)
                if self.api_key:
                    break

        if not self.api_key:
            env_vars = " 或 ".join(self._get_api_key_env_vars())
            raise ValueError(f"请设置 {env_vars} 环境变量")

    def _ensure_httpx(self):
        """确保 httpx 可用"""
        if not HAS_HTTPX:
            raise ImportError("请安装 httpx: pip install httpx")

    def _get_headers(self) -> dict:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list, **kwargs) -> dict:
        """构建请求负载"""
        return {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

    def _request_with_retry(self, url: str, payload: dict) -> dict:
        """带重试的请求"""
        self._ensure_httpx()

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"API 请求 (尝试 {attempt + 1}/{self.config.max_retries})")

                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.post(
                        url,
                        headers=self._get_headers(),
                        json=payload,
                    )
                    response.raise_for_status()
                    return response.json()

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"请求超时 (尝试 {attempt + 1}): {e}")
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"HTTP 错误 (尝试 {attempt + 1}): {e.response.status_code}")
                # 4xx 错误不重试
                if 400 <= e.response.status_code < 500:
                    raise
            except Exception as e:
                last_error = e
                logger.warning(f"请求异常 (尝试 {attempt + 1}): {e}")

            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)  # 指数退避
                logger.info(f"等待 {delay}s 后重试...")
                time.sleep(delay)

        raise last_error or Exception("请求失败")

    def chat(self, messages: list, **kwargs) -> LLMResponse:
        """调用 LLM API"""
        payload = self._build_payload(messages, **kwargs)
        url = f"{self.BASE_URL}/chat/completions"

        data = self._request_with_retry(url, payload)

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    def _get_analysis_prompt(self, task: str) -> dict:
        """获取分析任务的提示词"""
        return ANALYSIS_PROMPTS.get(task, ANALYSIS_PROMPTS["full_review"])

    def analyze_contract(self, contract_text: str, task: str = "full_review") -> str:
        """分析合同"""
        prompts = self._get_analysis_prompt(task)
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": f"{prompts['user']}\n\n合同内容：\n{contract_text}"},
        ]
        response = self.chat(messages)
        return response.content


class QwenClient(BaseLLMClient):
    """通义千问 API 客户端"""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        config: Optional[LLMConfig] = None,
    ):
        super().__init__(api_key, model, config)

    def _get_api_key_env_vars(self) -> list:
        return ["QWEN_API_KEY", "DASHSCOPE_API_KEY"]


class QwenVLClient(BaseLLMClient):
    """通义千问 VL 多模态客户端 (用于 OCR)"""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus",
        config: Optional[LLMConfig] = None,
    ):
        super().__init__(api_key, model, config)

    def _get_api_key_env_vars(self) -> list:
        return ["QWEN_API_KEY", "DASHSCOPE_API_KEY"]

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """将图片编码为 base64"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_mime_type(self, image_path: Union[str, Path]) -> str:
        """获取图片 MIME 类型"""
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(suffix, "image/jpeg")

    def ocr(self, image_path: Union[str, Path], prompt: Optional[str] = None) -> str:
        """对图片进行 OCR 识别"""
        image_base64 = self._encode_image(image_path)
        mime_type = self._get_image_mime_type(image_path)

        prompts = self._get_analysis_prompt("ocr")
        user_prompt = prompt or prompts["user"]

        messages = [
            {"role": "system", "content": prompts["system"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]

        response = self.chat(messages)
        return response.content


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API 客户端"""

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        config: Optional[LLMConfig] = None,
    ):
        super().__init__(api_key, model, config)

    def _get_api_key_env_vars(self) -> list:
        return ["DEEPSEEK_API_KEY"]


def extract_json_from_response(text: str) -> Optional[dict]:
    """从 LLM 响应中提取 JSON"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


class ContractReviewClient:
    """合同审核客户端 - 统一接口"""

    def __init__(
        self,
        provider: Literal["qwen", "deepseek"] = "deepseek",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        self.provider = provider
        self.config = config or LLMConfig()

        if provider == "qwen":
            self.client = QwenClient(
                api_key=api_key,
                model=model or "qwen-plus",
                config=self.config
            )
            # 同时初始化 VL 客户端用于 OCR
            self.vl_client = QwenVLClient(
                api_key=api_key,
                model="qwen-vl-plus",
                config=self.config
            )
        elif provider == "deepseek":
            self.client = DeepSeekClient(
                api_key=api_key,
                model=model or "deepseek-chat",
                config=self.config
            )
            self.vl_client = None
        else:
            raise ValueError(f"不支持的提供商: {provider}")

        logger.info(f"ContractReviewClient 初始化完成 (provider={provider})")

    def review(self, contract_text: str, task: str = "full_review") -> str:
        """审核合同"""
        logger.info(f"开始审核合同 (task={task}, 字符数={len(contract_text)})")
        return self.client.analyze_contract(contract_text, task)

    def extract_info(self, contract_text: str) -> dict:
        """提取合同关键信息"""
        logger.info("提取合同关键信息...")
        result = self.client.analyze_contract(contract_text, "clause_extract")

        # 尝试解析 JSON
        parsed = extract_json_from_response(result)
        if parsed:
            return parsed
        return {"raw_result": result}

    def analyze_risks(self, contract_text: str) -> str:
        """风险分析"""
        logger.info("执行风险分析...")
        return self.client.analyze_contract(contract_text, "risk_analysis")

    def check_compliance(self, contract_text: str) -> str:
        """合规性检查"""
        logger.info("执行合规性检查...")
        return self.client.analyze_contract(contract_text, "compliance_check")

    def ocr(self, image_path: Union[str, Path]) -> str:
        """对合同图片进行 OCR 识别"""
        if self.vl_client is None:
            # 如果当前 provider 不支持 VL，临时创建 Qwen VL 客户端
            logger.info("当前 provider 不支持 OCR，使用 Qwen-VL...")
            try:
                vl_client = QwenVLClient()
                return vl_client.ocr(image_path)
            except ValueError as e:
                raise ValueError(
                    f"OCR 需要 Qwen API，请设置 QWEN_API_KEY 环境变量: {e}"
                )

        logger.info(f"OCR 识别图片: {image_path}")
        return self.vl_client.ocr(image_path)


# 命令行接口
def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="合同审核 LLM 客户端")
    parser.add_argument("input", nargs="?", help="合同文本或文件路径")
    parser.add_argument(
        "--provider", "-p",
        choices=["qwen", "deepseek"],
        default="deepseek",
        help="LLM 提供商 (默认: deepseek)"
    )
    parser.add_argument(
        "--task", "-t",
        choices=["full_review", "risk_analysis", "clause_extract", "compliance_check"],
        default="full_review",
        help="分析任务 (默认: full_review)"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="对图片进行 OCR 识别"
    )
    parser.add_argument(
        "--model", "-m",
        help="指定模型名称"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="请求超时时间 (默认: 120s)"
    )

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        print("\n示例:")
        print("  # OCR 识别图片")
        print("  python llm_client.py contract.jpg --ocr")
        print("")
        print("  # 风险分析")
        print("  python llm_client.py contract.txt --task risk_analysis")
        print("")
        print("  # 使用 Qwen 模型")
        print("  python llm_client.py contract.txt -p qwen")
        return

    config = LLMConfig(timeout=args.timeout)

    try:
        client = ContractReviewClient(
            provider=args.provider,
            model=args.model,
            config=config
        )

        if args.ocr:
            # OCR 模式
            result = client.ocr(args.input)
            print("=" * 60)
            print("OCR 识别结果:")
            print("=" * 60)
            print(result)
        else:
            # 文本分析模式
            input_path = Path(args.input)
            if input_path.exists():
                with open(input_path, "r", encoding="utf-8") as f:
                    contract_text = f.read()
            else:
                contract_text = args.input

            result = client.review(contract_text, args.task)
            print("=" * 60)
            print(f"分析结果 (task={args.task}):")
            print("=" * 60)
            print(result)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
