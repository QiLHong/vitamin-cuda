# 从本目录运行，用 Nsight Compute 按核函数名筛选采集；-f 会覆盖同名报告。
ncu -k regex:"gemm|Kernel" --set full -o profile_res -f python test_for_ncu.py
