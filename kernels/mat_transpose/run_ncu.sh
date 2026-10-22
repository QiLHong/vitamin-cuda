# 使用 Nsight Compute 采集名称匹配 transpose 的核函数指标，结果写入 profile_res。
# 原命令引用 test_.py，当前目录仅有 test.py；运行前需核对入口，本次保持命令原样。
ncu -k regex:"transpose" --set full -o profile_res -f python test_.py
