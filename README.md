YOLO11n + TensorRT C++ 本地摄像头无人机检测
完整部署 README 说明文档基于 YOLO11、DUT-ANTI-UAV 无人机数据集、TensorRT 10.16、CUDA 12.8、RTX 5060 显卡实现 C++ 端本地摄像头实时无人机检测 + TensorRT 加速
一、项目说明
## 效果演示

<img width="2552" height="1596" alt="162f4c77f6233c3de075a69a7420817b" src="https://github.com/user-attachments/assets/0af293d5-577a-495d-8e0e-515fcb9a0070" />


## 目录

- [项目说明](#项目说明)
- [环境要求](#环境要求)
- [模型训练](#模型训练)
- [模型转换](#模型转换)
- [C++ 部署](#c-部署)
- [运行效果](#运行效果)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [参考与致谢](#参考与致谢)
- [许可证](#许可证)

## 项目说明

| 组件 | 选型 |
|------|------|
| 检测模型 | YOLO11n |
| 训练数据集 | [DUT-ANTI-UAV](https://github.com/wangdongdut/DUT-Anti-UAV) 无人机公开数据集 |
| 训练轮数 | 200 epochs |
| 推理加速 | TensorRT 10.16 + FP16 |
| 推理框架 | C++ + OpenCV 4.10.0 |
| GPU | NVIDIA RTX 5060 Laptop |

## 环境要求

### 硬件

- NVIDIA RTX 5060 Laptop GPU（其他 NVIDIA 显卡亦可，需调整 CUDA 算力）

### 软件

| 软件 | 版本 |
|------|------|
| Windows | 11 |
| CUDA | 12.8 |
| TensorRT | 10.16.0.72 |
| Visual Studio | 2022 |
| OpenCV | 4.10.0 |
| Python | ≥ 3.10（仅模型转换用） |

## 模型训练

模型训练已完成，使用 YOLO11n 在 DUT-ANTI-UAV 数据集上训练 200 轮，输出权重文件 `best.pt`。

> 训练代码可参考 [Ultralytics YOLO11](https://docs.ultralytics.com/zh/models/yolo11/) 官方文档。

## 模型转换

### 1. PT → ONNX

```python
# export_to_onnx.py
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, half=True)
运行后得到 best.onnx。

2. ONNX → TensorRT Engine
将 best.onnx 复制到 TensorRT 的 bin 目录：


# 进入 TensorRT bin 目录
cd D:\TensorRT-10.16.0.72\bin

# FP16 精度转换
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
看到终端输出 PASSED 即转换成功，得到 best.engine。

C++ 部署
1. 新建 Visual Studio 项目
项目名称：YOLO11_TensorRT_UAV
平台：x64
2. 添加源文件
yolo11.h — 推理类声明
yolo11.cpp — 推理类实现
main.cpp — 主程序（摄像头读取 + 推理循环）
3. 配置项目属性
VC++ 目录 → 包含目录：
<img width="2252" height="1233" alt="image" src="https://github.com/user-attachments/assets/0cf8b250-acbe-4a39-b789-1be4289bdeb8" />


D:\TensorRT-10.16.0.72\include
D:\opencv\build\include
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include
VC++ 目录 → 库目录：


D:\TensorRT-10.16.0.72\lib
D:\opencv\build\x64\vc16\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64
链接器 → 输入 → 附加依赖项：

<img width="1108" height="754" alt="image" src="https://github.com/user-attachments/assets/39b23c41-249b-4e50-84bb-09e96dd420ea" />

nvinfer_10.lib
nvinfer_plugin_10.lib
nvonnxparser_10.lib
cudart.lib
opencv_world4100d.lib   # Debug 模式
opencv_world4100.lib    # Release 模式
4. 文件放置（关键）
将以下文件复制到 x64\Debug\ 和 x64\Release\ 目录：

best.engine
opencv_world4100d.dll / opencv_world4100.dll
nvinfer_10.dll
确保 .exe、.dll、.engine 在同一目录下。

5. 编译运行

生成 → 清理解决方案
生成 → 生成解决方案
输出 生成成功：1 后按 F5 或 Ctrl + F5 运行，程序自动打开摄像头开始实时检测。

运行效果
摄像头实时画面，无人机目标自动框选
显示类别标签 UAV 及置信度
TensorRT FP16 低延迟推理
C++ 端高性能，适合实际部署
常见问题
问题	原因	解决方案
LNK1104 找不到 nvinfer.lib	库目录或库名错误	确保库目录指向 TensorRT\lib，库名带 _10 后缀
找不到 opencv_world4100d.dll	DLL 不在可搜索路径中	将 DLL 复制到 .exe 同级目录
加载 best.engine 失败	TensorRT 版本不一致	使用与生成 Engine 相同的 TensorRT 版本
摄像头打不开	权限不足或设备索引错误	以管理员运行；将代码中摄像头索引改为 0
项目结构

YOLO11_TensorRT_UAV/
├── yolo11.h                  # 推理类头文件
├── yolo11.cpp                # 推理类实现
├── main.cpp                  # 主程序入口
├── best.pt                   # 训练权重（模型转换用）
├── best.onnx                 # ONNX 中间模型
├── best.engine               # TensorRT 序列化引擎
├── export_to_onnx.py         # PT → ONNX 导出脚本
└── README.md
参考与致谢
Ultralytics YOLO11
DUT-ANTI-UAV 数据集
NVIDIA TensorRT 文档
CSDN 博客《TensorRT 加速 YOLOv11——C++ 端加速》
本项目为学习实践项目，在豆包、Gemini 等 AI 工具辅助下手动完成，仅供学习与参考。

许可证
本项目采用 MIT License。欢迎 Star & Fork。



---

有问题可以随时联系50333038@qq.com
