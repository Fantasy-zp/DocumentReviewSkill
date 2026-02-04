"""
合同审核 LLM 调用客户端
支持 Qwen 和 DeepSeek API
"""

import os
import json
from typing import Optional, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import httpx
except ImportError:
    httpx = None  # 可选依赖


@dataclass
class LLMResponse:
    """LLM 响应结果"""
    content: str
    model: str
    usage: dict
    raw_response: dict


class BaseLLMClient(ABC):
    """LLM 客户端基类"""

    @abstractmethod
    def chat(self, messages: list, **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def analyze_contract(self, contract_text: str, task: str) -> str:
        pass


class QwenClient(BaseLLMClient):
    """通义千问 API 客户端"""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",  # qwen-turbo, qwen-plus, qwen-max
    ):
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("请设置 QWEN_API_KEY 或 DASHSCOPE_API_KEY 环境变量")

    def chat(self, messages: list, **kwargs) -> LLMResponse:
        """调用 Qwen API"""
        if httpx is None:
            raise ImportError("请安装 httpx: pip install httpx")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=data.get("usage", {}),
            raw_response=data,
        )

    def analyze_contract(self, contract_text: str, task: str = "full_review") -> str:
        """分析合同"""
        prompts = self._get_analysis_prompt(task)
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": f"{prompts['user']}\n\n合同内容：\n{contract_text}"},
        ]
        response = self.chat(messages)
        return response.content


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API 客户端"""

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",  # deepseek-chat, deepseek-reasoner
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

    def chat(self, messages: list, **kwargs) -> LLMResponse:
        """调用 DeepSeek API"""
        if httpx is None:
            raise ImportError("请安装 httpx: pip install httpx")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=data.get("usage", {}),
            raw_response=data,
        )

    def analyze_contract(self, contract_text: str, task: str = "full_review") -> str:
        """分析合同"""
        prompts = self._get_analysis_prompt(task)
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": f"{prompts['user']}\n\n合同内容：\n{contract_text}"},
        ]
        response = self.chat(messages)
        return response.content


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
请从合同中提取以下关键信息，以 JSON 格式输出：
- contract_type: 合同类型
- parties: 合同双方信息
- amount: 合同金额
- duration: 合同期限
- key_obligations: 主要义务
- termination_conditions: 解除条件
- dispute_resolution: 争议解决方式""",
        "user": "请从以下合同中提取关键信息：",
    },
    "compliance_check": {
        "system": """你是一位法律合规专家。
请检查合同是否符合相关法律法规要求，包括：
1. 必备条款是否齐全
2. 是否存在违反强制性法规的条款
3. 格式条款是否履行了提示说明义务
4. 特殊事项是否需要审批备案""",
        "user": "请检查以下合同的合规性：",
    },
}


def _get_analysis_prompt(task: str) -> dict:
    """获取分析任务的提示词"""
    return ANALYSIS_PROMPTS.get(task, ANALYSIS_PROMPTS["full_review"])


# 为两个客户端类添加方法
QwenClient._get_analysis_prompt = staticmethod(_get_analysis_prompt)
DeepSeekClient._get_analysis_prompt = staticmethod(_get_analysis_prompt)


class ContractReviewClient:
    """合同审核客户端 - 统一接口"""

    def __init__(
        self,
        provider: Literal["qwen", "deepseek"] = "deepseek",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if provider == "qwen":
            self.client = QwenClient(api_key=api_key, model=model or "qwen-plus")
        elif provider == "deepseek":
            self.client = DeepSeekClient(api_key=api_key, model=model or "deepseek-chat")
        else:
            raise ValueError(f"不支持的提供商: {provider}")

        self.provider = provider

    def review(self, contract_text: str, task: str = "full_review") -> str:
        """审核合同"""
        return self.client.analyze_contract(contract_text, task)

    def extract_info(self, contract_text: str) -> dict:
        """提取合同关键信息"""
        result = self.client.analyze_contract(contract_text, "clause_extract")
        try:
            # 尝试解析 JSON
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_result": result}

    def analyze_risks(self, contract_text: str) -> str:
        """风险分析"""
        return self.client.analyze_contract(contract_text, "risk_analysis")

    def check_compliance(self, contract_text: str) -> str:
        """合规性检查"""
        return self.client.analyze_contract(contract_text, "compliance_check")


# 使用示例
if __name__ == "__main__":
    # 示例用法
    sample_contract = """
    合同编号：2024-001

    甲方：XX科技有限公司
    乙方：XX咨询有限公司

    第一条 服务内容
    乙方为甲方提供软件开发咨询服务...

    第二条 服务费用
    服务费用总计人民币10万元...
    """

    # 使用 DeepSeek
    # client = ContractReviewClient(provider="deepseek")
    # result = client.review(sample_contract)
    # print(result)

    print("合同审核 LLM 客户端已就绪")
    print("支持的提供商: qwen, deepseek")
    print("支持的任务: full_review, risk_analysis, clause_extract, compliance_check")
