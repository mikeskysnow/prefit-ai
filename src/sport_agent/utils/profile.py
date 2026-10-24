"""用户画像的 prompt 文本化。

user_profile 是给程序用的 dict（英文键名，含 user_id/source 等内部字段）。
直接塞进 prompt 时渲染成 `{'user_id': '1986...', 'age': 36, ..., 'source': 'db'}`，
LLM 需要在噪音里自己认字段，实测会出现「年龄未知」这类明明有数据却不用的情况。
本模块把画像转成中文可读的一行文本，只保留对研判有意义的字段。
"""

from typing import Any

_GENDER_CN = {"male": "男性", "female": "女性"}

_LEVEL_CN = {
    "beginner": "初学者",
    "intermediate": "有一定基础",
    "advanced": "高水平",
}

_GOAL_CN = {
    "health": "健康促进",
    "fat_loss": "减脂",
    "endurance": "耐力提升",
    "strength": "力量增强",
    "rehab": "康复",
}

_TIME_CN = {"morning": "早晨", "noon": "中午", "afternoon": "下午", "evening": "晚间"}

_CAPACITY_CN = {"low": "受限", "normal": "一般", "good": "良好"}


_SPORT_TYPE_CN = {
    "running": "跑步",
    "walk": "快走",
    "indoor_bike": "室内骑行",
    "fitness": "健身",
    "other": "其他",
}


def format_sport_record(state: dict[str, Any] | None) -> str:
    """把用户上传的运动记录转成 prompt 文本。

    与 format_medical_report 同样的取舍：没有上传时明确说明，
    避免 LLM 把空值当成「最近没运动」来用。
    """
    if not state or not state.get("sport_record_ready"):
        return "用户本次未上传运动记录（近期负荷请参考上方的连续训练天数）"

    rec = state.get("sport_record") or {}
    parts: list[str] = []

    sport = _SPORT_TYPE_CN.get(str(rec.get("sport_type")), rec.get("sport_type") or "未知")
    raw = rec.get("sport_type_raw")
    parts.append(f"运动类型 {sport}（原文「{raw}」）" if raw else f"运动类型 {sport}")

    if rec.get("start_time"):
        parts.append(f"时间 {rec['start_time']}")
    if rec.get("duration_sec"):
        parts.append(f"时长 {round(rec['duration_sec'] / 60)} 分钟")
    if rec.get("distance_m"):
        parts.append(f"距离 {round(float(rec['distance_m']) / 1000, 2)} 公里")
    if rec.get("avg_hr"):
        parts.append(f"平均心率 {rec['avg_hr']} bpm")
    if rec.get("max_hr"):
        parts.append(f"最大心率 {rec['max_hr']} bpm")
    if rec.get("avg_pace_sec_per_km"):
        pace = int(rec["avg_pace_sec_per_km"])
        parts.append(f"平均配速 {pace // 60}'{pace % 60:02d}\"/km")
    if rec.get("calories"):
        parts.append(f"热量 {rec['calories']} kcal")
    if rec.get("intensity_zone"):
        parts.append(f"心率区间 {rec['intensity_zone']}")
    if rec.get("training_effect"):
        parts.append(f"训练效果 {rec['training_effect']}")
    if rec.get("body_battery_delta") is not None:
        parts.append(f"身体电量变化 {rec['body_battery_delta']}")

    lines = ["、".join(parts)]

    # 运动画像四项：习惯运动、强度、过往能力、频率（作为制定计划的依据）
    profile_parts: list[str] = []
    if rec.get("habitual_sport"):
        profile_parts.append(f"习惯运动 {rec['habitual_sport']}")
    if rec.get("intensity_level"):
        level = _INTENSITY_CN.get(str(rec["intensity_level"]), rec["intensity_level"])
        profile_parts.append(f"本次强度 {level}")
    if rec.get("capacity_level"):
        cap = _LEVEL_CN.get(str(rec["capacity_level"]), rec["capacity_level"])
        profile_parts.append(f"运动能力档位 {cap}")
    if rec.get("frequency_hint"):
        profile_parts.append(f"运动频率 {rec['frequency_hint']}")
    if profile_parts:
        lines.append("运动画像：" + "、".join(profile_parts))

    if rec.get("capacity_assessment"):
        lines.append(f"过往运动能力分析：{rec['capacity_assessment']}")

    summary = state.get("sport_record_summary")
    if summary:
        lines.append(f"记录总结：{summary}")

    implications = state.get("sport_record_implications") or []
    if implications:
        lines.append(f"对今日计划的影响：{'；'.join(implications)}")

    return "\n".join(lines)


def format_medical_report(state: dict[str, Any] | None) -> str:
    """把体检报告分析结果转成 prompt 文本。

    与 format_profile 同样的取舍：没有报告时明确说明「未上传」，
    避免 LLM 把空值当成「体检一切正常」来用。
    """
    if not state or not state.get("medical_report_ready"):
        return "用户未上传体检报告（请勿据此假设其体检结果正常或异常）"

    parts: list[str] = []

    # 心血管放最前：直接决定强度上限，是运动安全性的第一约束
    report = state.get("medical_report") or {}
    conditions = report.get("cardiac_conditions") or []
    if conditions:
        risk = _CARDIAC_RISK_CN.get(str(report.get("cardiac_risk_level")))
        head = f"心血管病症表现（风险{risk}）：" if risk else "心血管病症表现："
        parts.append(head + "；".join(conditions))
    if limit := report.get("cardiac_exercise_limit"):
        parts.append(f"心血管运动限制：{limit}")

    findings = state.get("medical_report_findings") or []
    parts.append(f"关键异常发现：{'；'.join(findings)}" if findings else "关键异常发现：无")

    restrictions = state.get("medical_report_restrictions") or []
    if restrictions:
        parts.append(f"运动禁忌（硬约束）：{'；'.join(restrictions)}")

    advice = state.get("medical_report_advice")
    if advice:
        parts.append(f"报告运动建议：{advice}")

    return "\n".join(parts)


_INTENSITY_CN = {"low": "低强度", "mid": "中等强度", "high": "高强度"}

_CARDIAC_RISK_CN = {"none": "未见异常", "low": "轻度", "mid": "中度", "high": "重度"}


def _sport_profile_parts(profile: dict[str, Any]) -> list[str]:
    """运动记录推断出的运动画像片段（习惯运动/强度/能力/频率）。"""
    parts: list[str] = []

    if habit_sport := profile.get("habitual_sport"):
        parts.append(f"习惯运动 {habit_sport}")

    intensity = profile.get("last_intensity_level")
    if intensity:
        parts.append(f"最近一次运动强度 {_INTENSITY_CN.get(str(intensity), intensity)}")

    if assessment := profile.get("capacity_assessment"):
        parts.append(f"过往运动能力分析：{assessment}")

    if freq_hint := profile.get("frequency_hint"):
        parts.append(f"运动频率线索 {freq_hint}")

    return parts


# 临床指标的文本化规则：(画像键, 中文名, 单位)。分组顺序对应各组指标的作用，
# 让 LLM 能按「安全上限 → 代谢 → 耐力 → 恢复 → 动作选择」的顺序读到。
_CLINICAL_GROUPS: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    (
        "有氧能力与能量代谢",
        (
            ("hemoglobin", "血红蛋白", " g/L"),
            ("fasting_glucose", "空腹血糖", " mmol/L"),
            ("hba1c", "糖化血红蛋白", "%"),
        ),
    ),
    (
        "心血管风险与代谢负荷",
        (
            ("total_cholesterol", "总胆固醇", " mmol/L"),
            ("ldl_c", "LDL-C", " mmol/L"),
            ("hdl_c", "HDL-C", " mmol/L"),
            ("triglycerides", "甘油三酯", " mmol/L"),
            ("creatinine", "肌酐", " umol/L"),
            ("egfr", "eGFR", " mL/min/1.73m²"),
            ("electrolytes", "电解质", ""),
        ),
    ),
    (
        "有氧耐力",
        (
            ("spo2", "血氧饱和度", "%"),
            ("fev1_fvc", "FEV1/FVC", "%"),
            ("lung_function", "肺功能结论", ""),
        ),
    ),
    (
        "代谢率与恢复能力",
        (
            ("tsh", "TSH", " mIU/L"),
            ("thyroid_conclusion", "甲功结论", ""),
            ("testosterone", "睾酮", ""),
        ),
    ),
    (
        "运动类型与动作选择",
        (
            ("body_fat_pct", "体脂率", "%"),
            ("muscle_mass", "肌肉量", " kg"),
            ("bone_density_t", "骨密度T值", ""),
            ("bone_density_conclusion", "骨密度结论", ""),
            ("joint_condition", "关节与脊柱", ""),
        ),
    ),
)


def _clinical_parts(profile: dict[str, Any]) -> list[str]:
    """体检报告抽取的临床指标文本化，按对运动处方的作用分组。

    只输出报告确有的项：这里的缺失是「报告没做这项检查」，与 format_profile 顶部
    基础字段的「未提供」不同，不必逐项声明未知，否则 prompt 会被大量占位噪音淹没。
    """
    parts: list[str] = []

    # 心血管专项优先：直接决定强度上限
    conditions = profile.get("cardiac_conditions") or []
    if conditions:
        parts.append(f"心血管病症表现（{'；'.join(conditions)}）")
    cardiac_risk = _CARDIAC_RISK_CN.get(str(profile.get("cardiac_risk_level")))
    if cardiac_risk:
        parts.append(f"心血管风险 {cardiac_risk}")
    if limit := profile.get("cardiac_exercise_limit"):
        parts.append(f"心血管运动限制：{limit}")

    if ecg := profile.get("ecg_conclusion"):
        parts.append(f"心电图 {ecg}")

    for group_name, fields in _CLINICAL_GROUPS:
        items = [
            f"{cn} {profile[key]}{unit}"
            for key, cn, unit in fields
            if profile.get(key) is not None and profile.get(key) != ""
        ]
        if items:
            parts.append(f"{group_name}（{'、'.join(items)}）")

    # 维生素D 的单位随报告变（ng/mL 或 nmol/L），不能写死
    vit_d = profile.get("vitamin_d")
    if vit_d is not None:
        unit = profile.get("vitamin_d_unit") or ""
        parts.append(f"维生素D {vit_d}{(' ' + unit) if unit else ''}")

    return parts


def format_profile(profile: dict[str, Any] | None) -> str:
    """把 user_profile 转成中文可读文本供 prompt 使用。

    缺失或无意义的值（如 resting_hr=0）写成「未提供」而不是省略，
    让 LLM 明确区分「没这项数据」和「我没注意到这项数据」。
    """
    if not profile:
        return "画像数据缺失（无年龄、性别、病史等信息，请勿凭空假设）"

    parts: list[str] = []

    age = profile.get("age")
    parts.append(f"年龄 {int(age)} 岁" if age else "年龄未提供")

    gender = _GENDER_CN.get(str(profile.get("gender")))
    parts.append(f"性别 {gender}" if gender else "性别未提供")

    height, weight, bmi = profile.get("height"), profile.get("weight"), profile.get("bmi")
    if height:
        parts.append(f"身高 {height} cm")
    if weight:
        parts.append(f"体重 {weight} kg")
    if bmi:
        parts.append(f"BMI {bmi}")

    resting_hr = profile.get("resting_hr")
    parts.append(f"静息心率基线 {resting_hr} bpm" if resting_hr else "静息心率基线未提供")

    level = profile.get("level")
    parts.append(f"运动水平 {_LEVEL_CN.get(str(level), level)}" if level else "运动水平未提供")

    goal = profile.get("goal")
    if goal:
        parts.append(f"运动目标 {_GOAL_CN.get(str(goal), goal)}")

    bp = profile.get("blood_pressure")
    if bp:
        parts.append(f"血压 {bp} mmHg")

    diseases = profile.get("history_disease") or []
    parts.append(f"既往病史 {'、'.join(diseases)}" if diseases else "既往病史 无")

    habit = profile.get("habit") or {}
    freq, pref_time = habit.get("freq_per_week"), habit.get("pref_time")
    if freq:
        parts.append(f"每周运动 {freq} 次")
    if pref_time:
        parts.append(f"习惯时段 {_TIME_CN.get(str(pref_time), pref_time)}")

    # 经常运动时长：从历史运动记录推导，作为计划时长的主要依据
    habitual_dur = profile.get("habitual_duration")
    if habitual_dur:
        parts.append(f"经常运动时长 {habitual_dur} 分钟")

    # 自定义运动能力数据（用户提供时覆盖自动采集的统计）
    if profile.get("avg_duration_7d"):
        parts.append(f"近7天平均运动时长 {profile['avg_duration_7d']} 分钟")
    if profile.get("avg_calories"):
        parts.append(f"平均运动消耗 {profile['avg_calories']} kcal")
    if profile.get("common_sport_type"):
        parts.append(f"常用运动类型 {profile['common_sport_type']}")
    if profile.get("avg_distance_km"):
        parts.append(f"平均运动距离 {profile['avg_distance_km']} km")
    if profile.get("avg_pace"):
        parts.append(f"平均配速 {profile['avg_pace']}")

    capacity = _CAPACITY_CN.get(str(profile.get("exercise_capacity")))
    if capacity:
        parts.append(f"运动能力 {capacity}")

    # 运动记录推断出的运动画像（习惯运动/上次强度/能力评估/频率）
    parts.extend(_sport_profile_parts(profile))

    # 体检报告抽取的临床指标，按对运动处方的作用分组
    parts.extend(_clinical_parts(profile))

    # 体检报告导出的禁忌必须进 prompt：这是运动处方的硬约束，漏掉可能造成伤害
    restrictions = profile.get("exercise_restrictions") or []
    if restrictions:
        parts.append(f"运动禁忌 {'；'.join(restrictions)}")

    systems = profile.get("report_systems") or {}
    if systems:
        detail = "；".join(f"{k}：{v}" for k, v in systems.items())
        parts.append(f"体检分项结论（{detail}）")

    return "，".join(parts)
