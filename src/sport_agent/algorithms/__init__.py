"""算法引擎层：确定性生理算法（纯 Python，可复现、可单元测试）。

对应文档《身体电量方案.md》Layer 1：
- LLM 不做数值计算，电量分数必须 100% 可复现。
- 算法参考 HRV4Training（Marco Altini 开源方案），与 Garmin Body Battery 相关性 r=0.82。
"""
