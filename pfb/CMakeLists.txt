cmake_minimum_required(VERSION 3.10)
project(ONNXInference)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV（系统已安装）
find_package(OpenCV REQUIRED)

# 设置 ONNX Runtime 路径（从环境变量读取）
set(ONNXRUNTIME_DIR "/home/wangpp/xm/onnxruntime-linux-x64-1.8.1")
set(ONNXRUNTIME_INCLUDE_DIR $ENV{ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIB_DIR $ENV{ONNXRUNTIME_DIR}/lib)

# 添加可执行文件
add_executable(onnx_inference zhuanhua.cpp)

# 链接 OpenCV
target_include_directories(onnx_inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_inference ${OpenCV_LIBS})

# 链接 ONNX Runtime
target_include_directories(onnx_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_directories(onnx_inference PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(onnx_inference onnxruntime)

# 设置运行时库搜索路径（让可执行文件能找到 .so）
set_target_properties(onnx_inference PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
    INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
)
