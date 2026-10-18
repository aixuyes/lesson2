# lesson2 装甲板识别 Demo
基于 Zenoh + OpenCV4 的图像发布订阅与装甲板灯条识别示例。
pub 读取视频文件通过 zenoh 发布图像流；sub 订阅图像流，完成灯条检测、灯条配对、装甲板候选框绘制。

## 环境依赖
- C++17
- CMake >=3.16
- zenohc 1.10.1
- zenohcxx 1.10.1
- OpenCV 4 (core imgproc imgcodecs highgui)
- Threads

> 启用 zenoh shared‑memory（共享内存），用于高速图像传输。

## 文件说明
| 文件 | 作用 |
|---|---|
| `pub.cpp` | 发布端：读取`raw.mp4`视频，封装自定义二进制包，通过zenoh发布图像流 |
| `sub.cpp` | 订阅端：接收图像，灯条检测+装甲配对，OpenCV窗口可视化显示结果 |
| `CMakeLists.txt` | CMake编译配置 |
| `raw.mp4` | 待播放的视频文件，请放置在可执行文件运行目录 |

### 自定义数据包格式（24字节头部 + BGR像素数据）
