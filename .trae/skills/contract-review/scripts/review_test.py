"""
合同审核测试脚本
用于测试文档解析和 LLM 审核流程

版本: 1.1.0
更新: 2025-02-04
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from doc_parser import DocumentParser, parse_with_info
from llm_client import ContractReviewClient, QwenVLClient, LLMConfig


def test_parse_only(file_path: str, use_ocr: bool = False) -> dict:
    """仅测试文档解析"""
    print(f"\n{'='*60}")
    print(f"测试文档解析: {file_path}")
    print('='*60)

    # 准备 OCR 回调
    ocr_callback = None
    if use_ocr:
        try:
            vl_client = QwenVLClient()
            ocr_callback = lambda p: vl_client.ocr(p)
            print("OCR 功能已启用")
        except ValueError as e:
            print(f"⚠️ OCR 不可用: {e}")

    try:
        info = parse_with_info(file_path, ocr_callback=ocr_callback)
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
    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
    except ValueError as e:
        print(f"❌ 不支持的文件: {e}")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        logger.exception("解析异常")
    return None


def test_full_review(file_path: str, provider: str = "deepseek", use_ocr: bool = False):
    """完整审核测试"""
    print(f"\n{'='*60}")
    print(f"完整审核测试: {file_path}")
    print(f"使用模型: {provider}")
    print('='*60)

    # 1. 解析文档
    print("\n[1/4] 解析文档...")

    ocr_callback = None
    if use_ocr or Path(file_path).suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
        try:
            vl_client = QwenVLClient()
            ocr_callback = lambda p: vl_client.ocr(p)
            print("   OCR 功能已启用")
        except ValueError as e:
            print(f"   ⚠️ OCR 不可用: {e}")
            if Path(file_path).suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
                print("   ❌ 图片文件需要 OCR 功能")
                return

    try:
        info = parse_with_info(file_path, ocr_callback=ocr_callback)
        print(f"   ✅ 解析成功 - {info['char_count']} 字符")
    except Exception as e:
        print(f"   ❌ 解析失败: {e}")
        return

    # 2. 初始化 LLM 客户端
    print(f"\n[2/4] 初始化 {provider} 客户端...")
    try:
        config = LLMConfig(timeout=180)  # 长文本可能需要更长时间
        client = ContractReviewClient(provider=provider, config=config)
        print("   ✅ 客户端初始化成功")
    except ValueError as e:
        print(f"   ❌ 客户端初始化失败: {e}")
        print("   请检查环境变量是否设置:")
        if provider == "deepseek":
            print("   - DEEPSEEK_API_KEY")
        else:
            print("   - QWEN_API_KEY 或 DASHSCOPE_API_KEY")
        return

    # 3. 提取关键信息
    print("\n[3/4] 提取合同关键信息...")
    try:
        extracted = client.extract_info(info["content"])
        print("   ✅ 信息提取完成")
        if isinstance(extracted, dict) and "raw_result" not in extracted:
            for k, v in extracted.items():
                print(f"      {k}: {v}")
    except Exception as e:
        print(f"   ⚠️ 信息提取异常: {e}")

    # 4. 风险审核
    print("\n[4/4] 执行风险审核...")
    try:
        review_result = client.analyze_risks(info["content"])
        print("   ✅ 风险审核完成")
        print("\n" + "="*60)
        print("审核结果:")
        print("="*60)
        print(review_result)
    except Exception as e:
        print(f"   ❌ 风险审核失败: {e}")
        logger.exception("审核异常")
        return

    # 保存结果
    output_dir = Path(file_path).parent / "review_results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"{Path(file_path).stem}_审核报告_{timestamp}.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# {info['filename']} 审核报告\n\n")
        f.write(f"- 审核时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 使用模型: {provider}\n")
        f.write(f"- 文档字符数: {info['char_count']}\n\n")
        f.write("---\n\n")
        f.write(review_result)

    print(f"\n📄 报告已保存: {output_file}")


def test_all_examples(examples_dir: str = None, provider: str = "deepseek", parse_only: bool = True):
    """测试所有示例文件"""
    # 默认使用项目根目录下的 examples
    if examples_dir is None:
        examples_dir = SCRIPT_DIR.parent.parent.parent.parent / "examples"

    examples_path = Path(examples_dir)

    if not examples_path.exists():
        # 尝试相对路径
        examples_path = Path("examples")
        if not examples_path.exists():
            print(f"❌ 目录不存在: {examples_dir}")
            return

    # 支持的文件扩展名
    supported_extensions = (".pdf", ".docx", ".txt", ".md")
    files = list(examples_path.glob("*"))
    supported = [f for f in files if f.suffix.lower() in supported_extensions]

    print(f"\n找到 {len(supported)} 个支持的文件:")
    for f in supported:
        print(f"  - {f.name}")

    for file_path in supported:
        if parse_only:
            test_parse_only(str(file_path))
        else:
            test_full_review(str(file_path), provider=provider)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="合同审核测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅解析文档
  python review_test.py examples/租赁合同.pdf --parse-only

  # 完整审核 (需要 API Key)
  python review_test.py examples/租赁合同.pdf -p deepseek

  # 测试所有示例文件的解析
  python review_test.py --all --parse-only

  # OCR 识别图片
  python review_test.py contract.jpg --ocr --parse-only

环境变量:
  DEEPSEEK_API_KEY    DeepSeek API 密钥
  QWEN_API_KEY        Qwen API 密钥 (用于 OCR)
"""
    )

    parser.add_argument("file", nargs="?", help="要审核的文件路径")
    parser.add_argument(
        "--provider", "-p",
        choices=["qwen", "deepseek"],
        default="deepseek",
        help="LLM 提供商 (默认: deepseek)"
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="仅测试解析，不调用 LLM"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="启用 OCR 功能 (用于图片文件)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="测试 examples 目录下所有文件"
    )
    parser.add_argument(
        "--examples-dir",
        default=None,
        help="指定 examples 目录路径"
    )

    args = parser.parse_args()

    if args.all:
        test_all_examples(
            examples_dir=args.examples_dir,
            provider=args.provider,
            parse_only=args.parse_only
        )
    elif args.file:
        if args.parse_only:
            test_parse_only(args.file, use_ocr=args.ocr)
        else:
            test_full_review(args.file, provider=args.provider, use_ocr=args.ocr)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
