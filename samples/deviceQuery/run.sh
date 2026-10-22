# 从本目录配置、编译并运行设备查询示例。
# 保留原命令；nproc 在此是 shell 变量，若要指定并行数应先为它赋值。
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$nproc
./build/deviceQuery
