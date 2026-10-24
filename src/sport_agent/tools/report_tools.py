"""上传件工具：体检报告 PDF 文本提取 + 运动记录图片编码。

设计要点：
- 只做「取原料」这一件事，语义理解交给 LLM，避免为各家体检机构的排版写死规则。
- 图片走多模态 LLM 直接识别，不引入 OCR 模型：手表运动记录截图是图标 + 数字的
  混排版式，OCR 只能拿到零散字符，仍需 LLM 还原字段归属，多一层反而丢信息。
- 体检报告页数多、且常含扫描图片页（实测样例 28 页里 12 页无文字层），
  因此按页累加并在超过预算时截断，防止 prompt 超长。
- 报告正文里真正影响运动处方的是「主检报告 / 医师建议 / 异常项」，
  这些页在正文靠前，截断优先保留前部页面是安全的。
"""

from __future__ import annotations

import base64
from pathlib import Path

from sport_agent.utils import get_logger

logger = get_logger(__name__)

# 单次送入 LLM 的报告文本上限（字符）。样例报告全文 8k 出头，2 万足够容纳，
# 又不至于让异常报告把 prompt 顶爆。
MAX_REPORT_CHARS = 20000

# 支持的图片格式（运动记录截图/拍照）
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# 图片体积上限（字节）。手表截图通常几百 KB，10MB 足够；再大 base64 后
# 会把请求体顶到多模态接口的上限。
MAX_IMAGE_BYTES = 10 * 1024 * 1024

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class ReportExtractError(Exception):
    """上传件不可解析（文件缺失、格式不支持、PDF 无文字层）。"""


def detect_upload_kind(path: str) -> str:
    """按后缀判断上传件类型：pdf / image。

    只判容器格式，内容是体检报告还是运动记录由 LLM 识别。

    Raises:
        ReportExtractError: 后缀既不是 pdf 也不在支持的图片格式内
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    raise ReportExtractError(
        f"不支持的文件格式 {suffix or '未知'}，仅支持 pdf 与图片（{'/'.join(sorted(IMAGE_SUFFIXES))}）"
    )


def encode_image_data_url(path: str) -> str:
    """把图片读成 data URL，供多模态 LLM 的 image_url 内容块使用。

    用 data URL 而非公网 URL：图片存在服务端本地，没有对外可访问地址，
    且运动记录含个人数据，不宜为了识别再暴露一个公开链接。

    Raises:
        ReportExtractError: 文件不存在、格式不支持或超出体积上限
    """
    f = Path(path)
    if not f.exists():
        raise ReportExtractError(f"图片文件不存在：{path}")

    suffix = f.suffix.lower()
    mime = _MIME_BY_SUFFIX.get(suffix)
    if not mime:
        raise ReportExtractError(f"不支持的图片格式 {suffix or '未知'}")

    raw = f.read_bytes()
    if not raw:
        raise ReportExtractError("图片内容为空")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ReportExtractError(
            f"图片超过 {MAX_IMAGE_BYTES // (1024 * 1024)}MB 上限，请压缩后重试"
        )

    encoded = base64.b64encode(raw).decode("ascii")
    logger.info("report.image_encoded", path=f.name, bytes=len(raw), mime=mime)
    return f"data:{mime};base64,{encoded}"


def extract_pdf_text(path: str, max_chars: int = MAX_REPORT_CHARS) -> str:
    """提取 PDF 全文文本。

    Args:
        path: PDF 绝对/相对路径
        max_chars: 文本预算，超出即截断

    Returns:
        拼接后的报告文本

    Raises:
        ReportExtractError: 文件不存在、后缀不是 .pdf、依赖缺失或全篇无文字层
    """
    f = Path(path)
    if not f.exists():
        raise ReportExtractError(f"报告文件不存在：{path}")
    if f.suffix.lower() != ".pdf":
        raise ReportExtractError(f"v1.0.0 仅支持 pdf 格式，当前为 {f.suffix or '未知'}")

    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ReportExtractError("缺少 pypdf 依赖，无法解析 PDF 报告") from e

    try:
        reader = PdfReader(str(f))
    except Exception as e:  # noqa: BLE001 —— pypdf 对损坏文件抛的异常类型不固定
        raise ReportExtractError(f"PDF 打开失败：{e}") from e

    chunks: list[str] = []
    total = 0
    truncated = False
    for page in reader.pages:
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 —— 个别页解析失败不该毁掉整份报告
            continue
        if not text:
            continue  # 扫描图片页无文字层，跳过
        if total + len(text) > max_chars:
            chunks.append(text[: max_chars - total])
            truncated = True
            break
        chunks.append(text)
        total += len(text)

    if not chunks:
        raise ReportExtractError("报告无可提取文字（可能是纯扫描件，需 OCR 支持）")

    # join 的换行符不在逐页预算内，最终再裁一次保证严格不超预算
    text_all = "\n".join(chunks)
    if len(text_all) > max_chars:
        text_all = text_all[:max_chars]
        truncated = True
    total = len(text_all)

    logger.info(
        "report.pdf_extracted",
        path=f.name,
        pages=len(reader.pages),
        chars=total,
        truncated=truncated,
    )
    return text_all
