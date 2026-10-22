# 使用 Nsight Compute 对匹配名称的核函数采集完整指标，原命令使用 -f 覆盖同名报告。
# 在本目录运行，并确保 ncu、CUDA 环境与 Python 依赖可用；本次未执行采集。
ncu -k regex:"fmha|flash" --set full -o profile_res -f python test_for_ncu.py
