"""
合同审核测试脚本
用于测试文档解析和 LLM 审核流程
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from doc_parser import DocumentParser, parse_with_info
from llm_client import ContractReviewClient


def test_parse_only(file_path: str):
    """仅测试文档解析"""
    print(f"\n{'='*60}")
    print(f"测试文档解析: {file_path}")
    print('='*60)

    try:
        info = parse_with_info(file_path)
        print(f"✅ 解析成功")
        print(f"   文件名: {info['filename']}")
        print(f"   类型: {info['file_type']}")
        print(f"   页数: {info['page_count']}")
        print(f"   字符数: {info['char_count']}")
        print(f"\n   内容预览 (前 500 字):")
        print("-" * 40)
        print(info["content"][:500])
        print("-" * 40)
        return info
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return None


def test_full_review(file_path: str, provider: str = "deepseek"):
    """完整审核测试"""
    print(f"\n{'='*60}")
    print(f"完整审核测试: {file_path}")
    print(f"使用模型: {provider}")
    print('='*60)

    # 1. 解析文档
    print("\n[1/4] 解析文档...")
    try:
        info = parse_with_info(file_path)
        print(f"✅ 解析成功 - {info['char_count']} 字符")
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return

    # 2. 初始化 LLM 客户端
    print(f"\n[2/4] 初始化 {provider} 客户端...")
    try:
        client = ContractReviewClient(provider=provider)
        print("✅ 客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        print("   请检查环境变量是否设置:")
        print("   - DEEPSEEK_API_KEY")
        print("   - QWEN_API_KEY 或 DASHSCOPE_API_KEY")
        return

    # 3. 提取关键信息
    print("\n[3/4] 提取合同关键信息...")
    try:
        extracted = client.extract_info(info["content"])
        print("✅ 信息提取完成")
        if isinstance(extracted, dict) and "raw_result" not in extracted:
            for k, v in extracted.items():
                print(f"   {k}: {v}")
    except Exception as e:
        print(f"⚠️ 信息提取异常: {e}")

    # 4. 风险审核
    print("\n[4/4] 执行风险审核...")
    try:
        review_result = client.analyze_risks(info["content"])
        print("✅ 风险审核完成")
        print("\n" + "="*60)
        print("审核结果:")
        print("="*60)
        print(review_result)
    except Exception as e:
        print(f"❌ 风险审核失败: {e}")
        return

    # 保存结果
    output_dir = Path(file_path).parent / "review_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{Path(file_path).stem}_审核报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# {info['filename']} 审核报告\n\n")
        f.write(f"审核时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"使用模型: {provider}\n\n")
        f.write("---\n\n")
        f.write(review_result)

    print(f"\n📄 报告已保存: {output_file}")


def test_all_examples(examples_dir: str = "examples", provider: str = "deepseek"):
    """测试所有示例文件"""
    examples_path = Path(examples_dir)

    if not examples_path.exists():
        print(f"❌ 目录不存在: {examples_dir}")
        return

    files = list(examples_path.glob("*"))
    supported = [f for f in files if f.suffix.lower() in (".pdf", ".docx", ".doc", ".txt")]

    print(f"\n找到 {len(supported)} 个支持的文件:")
    for f in supported:
        print(f"  - {f.name}")

    for file_path in supported:
        test_parse_only(str(file_path))


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="合同审核测试工具")
    parser.add_argument("file", nargs="?", help="要审核的文件路径")
    parser.add_argument("--provider", "-p", choices=["qwen", "deepseek"], default="deepseek", help="LLM 提供商")
    parser.add_argument("--parse-only", action="store_true", help="仅测试解析，不调用 LLM")
    parser.add_argument("--all", "-a", action="store_true", help="测试 examples 目录下所有文件")

    args = parser.parse_args()

    if args.all:
        test_all_examples(provider=args.provider)
    elif args.file:
        if args.parse_only:
            test_parse_only(args.file)
        else:
            test_full_review(args.file, provider=args.provider)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python review_test.py examples/租赁合同.pdf --parse-only")
        print("  python review_test.py examples/租赁合同.pdf -p deepseek")
        print("  python review_test.py --all --parse-only")


if __name__ == "__main__":
    main()
