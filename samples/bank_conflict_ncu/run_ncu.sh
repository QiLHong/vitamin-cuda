# 在本目录运行；这些指标分别观察共享内存指令、wavefront 和 bank 冲突计数。
# 保留原指标名及续行；能否采集取决于 GPU、Nsight Compute 版本和性能计数器权限。
ncu -k regex:"load" \
--metrics \
smsp__inst_executed_op_shared_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
-f python test_for_ncu.py
