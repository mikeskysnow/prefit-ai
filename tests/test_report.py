"""上传件（体检报告 / 运动记录图片）功能单元测试（不依赖 LLM）。

覆盖：
- PDF 文本提取的边界情况（文件不存在 / 非 pdf / 真实样例）
- 上传件判型与图片编码（后缀识别、空文件、超限、data URL 前缀）
- 画像合并规则（报告优先、None 不覆盖、BMI 重算、禁忌与病史并入）
- 运动记录落库行（幂等键、单位透传）
- 条件路由分流（上传 → report_analyze，未上传 → health_collect）
- prompt 文本化（未上传 / 已上传两态）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sport_agent.agents.schemas import MedicalReportAnalysis, SportRecordAnalysis
from sport_agent.nodes.report import _merge_profile, _record_to_row, route_after_memory_load
from sport_agent.tools.report_tools import (
    ReportExtractError,
    detect_upload_kind,
    encode_image_data_url,
    extract_pdf_text,
)
from sport_agent.utils import format_medical_report, format_profile, format_sport_record

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "docs" / "体检报告91011021.pdf"


# ===== PDF 提取 =====


def test_extract_missing_file():
    with pytest.raises(ReportExtractError, match="不存在"):
        extract_pdf_text("/tmp/definitely_not_here_9527.pdf")


def test_extract_rejects_non_pdf(tmp_path):
    f = tmp_path / "report.docx"
    f.write_text("体检报告")
    with pytest.raises(ReportExtractError, match="仅支持 pdf"):
        extract_pdf_text(str(f))


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="样例报告不存在")
def test_extract_sample_report():
    """样例报告含扫描页，验证有文字层的页能正常提取。"""
    text = extract_pdf_text(str(SAMPLE_PDF))
    assert len(text) > 1000
    assert "心" in text  # 体检报告必然含心率/心电图相关字样


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="样例报告不存在")
def test_extract_respects_budget():
    text = extract_pdf_text(str(SAMPLE_PDF), max_chars=500)
    assert len(text) <= 500


# ===== 上传件判型与图片编码 =====


def test_detect_upload_kind():
    assert detect_upload_kind("/tmp/a.pdf") == "pdf"
    assert detect_upload_kind("/tmp/a.PNG") == "image"
    assert detect_upload_kind("/tmp/a.jpeg") == "image"
    with pytest.raises(ReportExtractError, match="不支持的文件格式"):
        detect_upload_kind("/tmp/a.docx")


def test_encode_image_missing_file():
    with pytest.raises(ReportExtractError, match="不存在"):
        encode_image_data_url("/tmp/definitely_not_here_9527.png")


def test_encode_image_rejects_empty(tmp_path):
    f = tmp_path / "shot.png"
    f.write_bytes(b"")
    with pytest.raises(ReportExtractError, match="为空"):
        encode_image_data_url(str(f))


def test_encode_image_rejects_bad_suffix(tmp_path):
    f = tmp_path / "shot.bmp"
    f.write_bytes(b"\x00\x01")
    with pytest.raises(ReportExtractError, match="不支持的图片格式"):
        encode_image_data_url(str(f))


def test_encode_image_data_url(tmp_path):
    f = tmp_path / "shot.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    url = encode_image_data_url(str(f))
    assert url.startswith("data:image/png;base64,")


def test_encode_image_rejects_oversize(tmp_path, monkeypatch):
    from sport_agent.tools import report_tools

    monkeypatch.setattr(report_tools, "MAX_IMAGE_BYTES", 16)
    f = tmp_path / "shot.jpg"
    f.write_bytes(b"x" * 64)
    with pytest.raises(ReportExtractError, match="上限"):
        report_tools.encode_image_data_url(str(f))


# ===== 运动记录落库行 =====


def test_record_to_row_idempotent_key(tmp_path):
    """同一张图两次上传必须得到同一个 raw_session_id，否则历史里会堆重复记录。"""
    f = tmp_path / "garmin.png"
    f.write_bytes(b"same-bytes")
    rec = SportRecordAnalysis(
        sport_type="running", duration_sec=2535, distance_m=5200.0, avg_hr=142, summary="5.2 公里"
    )

    row_a = _record_to_row(str(f), rec)
    row_b = _record_to_row(str(f), rec)

    assert row_a["raw_session_id"] == row_b["raw_session_id"]
    assert row_a["raw_session_id"].startswith("img_")
    assert row_a["duration_sec"] == 2535
    assert row_a["distance_m"] == 5200.0
    assert row_a["avg_hr"] == 142
    # 图片没有结束时间，缺失字段按训练历史表的默认值填
    assert row_a["end_time"] is None
    assert row_a["calories"] == 0


def test_record_to_row_differs_by_content(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    rec = SportRecordAnalysis(sport_type="walk", summary="快走")

    assert _record_to_row(str(a), rec)["raw_session_id"] != _record_to_row(str(b), rec)[
        "raw_session_id"
    ]


# ===== 画像合并 =====


def _analysis(**kw) -> MedicalReportAnalysis:
    kw.setdefault("exercise_advice", "建议中低强度有氧")
    return MedicalReportAnalysis(**kw)


def test_merge_report_overrides_app_profile():
    """报告是医疗机构实测，优先于 APP 侧自填/设备估算值。"""
    profile = {"age": 30, "height": 170.0, "weight": 80.0, "resting_hr": 70}
    merged = _merge_profile(profile, _analysis(age=45, weight=72.0, resting_hr=52))

    assert merged["age"] == 45
    assert merged["weight"] == 72.0
    assert merged["resting_hr"] == 52
    assert merged["height"] == 170.0  # 报告未给，保留原值
    assert merged["bmi"] == 24.9  # 72 / 1.7^2，用合并后的体重重算
    assert merged["profile_source"] == "medical_report"


def test_merge_keeps_original_when_report_empty():
    profile = {"age": 30, "gender": "male", "height": 175.0, "weight": 70.0}
    merged = _merge_profile(profile, _analysis())

    assert merged["age"] == 30
    assert merged["gender"] == "male"


def test_merge_blood_pressure_and_findings():
    merged = _merge_profile(
        {"history_disease": ["高血压"]},
        _analysis(
            systolic_bp=103,
            diastolic_bp=57,
            key_findings=["心动过缓 52次/分", "高血压"],
            exercise_restrictions=["避免颈部大幅度负重动作"],
            exercise_capacity="normal",
            cardiovascular="窦性心动过缓",
        ),
    )

    assert merged["blood_pressure"] == "103/57"
    # 关键发现并入病史且不重复已有项
    assert merged["history_disease"] == ["高血压", "心动过缓 52次/分"]
    assert merged["exercise_restrictions"] == ["避免颈部大幅度负重动作"]
    assert merged["exercise_capacity"] == "normal"
    assert merged["report_systems"]["心血管"] == "窦性心动过缓"


def test_merge_does_not_mutate_input():
    profile = {"age": 30}
    _merge_profile(profile, _analysis(age=45))
    assert profile["age"] == 30


# ===== 路由 =====


def test_route_skips_report_when_not_uploaded():
    assert route_after_memory_load({}) == "health_collect"
    assert route_after_memory_load({"report_path": ""}) == "health_collect"


def test_route_enters_report_when_uploaded():
    assert route_after_memory_load({"report_path": "/tmp/a.pdf"}) == "report_analyze"


# ===== prompt 文本化 =====


def test_format_report_not_uploaded():
    text = format_medical_report({})
    assert "未上传" in text


def test_format_report_ready():
    text = format_medical_report(
        {
            "medical_report_ready": True,
            "medical_report_findings": ["心动过缓 52次/分"],
            "medical_report_restrictions": ["避免颈部负重"],
            "medical_report_advice": "建议中低强度有氧",
        }
    )
    assert "心动过缓 52次/分" in text
    assert "避免颈部负重" in text
    assert "建议中低强度有氧" in text


def test_format_sport_record_not_uploaded():
    text = format_sport_record({})
    assert "未上传" in text


def test_format_sport_record_ready():
    """单位换算要落在 prompt 文本上：秒→分钟、米→公里、秒/km→m'ss"。"""
    text = format_sport_record(
        {
            "sport_record_ready": True,
            "sport_record": {
                "sport_type": "running",
                "sport_type_raw": "户外跑步",
                "duration_sec": 2535,
                "distance_m": 5200.0,
                "avg_hr": 142,
                "avg_pace_sec_per_km": 390,
                "calories": 320,
                "body_battery_delta": -18,
            },
            "sport_record_summary": "户外跑步 5.20 公里，用时 42:15",
            "sport_record_implications": ["昨日中等强度 42 分钟，今日宜降量"],
        }
    )
    assert "跑步" in text
    assert "户外跑步" in text
    assert "42 分钟" in text
    assert "5.2 公里" in text
    assert "142 bpm" in text
    assert "6'30\"/km" in text
    assert "身体电量变化 -18" in text
    assert "今日宜降量" in text


def test_format_profile_includes_report_fields():
    """报告导出的禁忌与分项结论必须进 prompt，否则计划生成看不到硬约束。"""
    text = format_profile(
        {
            "age": 45,
            "gender": "male",
            "blood_pressure": "103/57",
            "exercise_capacity": "low",
            "exercise_restrictions": ["避免颈部大幅度负重动作"],
            "report_systems": {"心血管": "窦性心动过缓"},
        }
    )
    assert "103/57" in text
    assert "受限" in text
    assert "避免颈部大幅度负重动作" in text
    assert "窦性心动过缓" in text
