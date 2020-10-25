# 黑盒AUTOML
本项目使用AUTOML方法在指定搜索空间中尝试对黑盒结构进行重构

算法流程见`docs/算法流程.docx`

## 目录结构
```
├── config                    # 配置文件目录
├── docker                    # 构建docker镜像的目录
├── docs                      # 放置项目文档的目录，如接口api文档、需求文档等
├── auto_ml                    # 项目主体代码目录，该目录下的子目录仅供参考，根据实际情况调整
│  ├── __init__.py            # 主体代码的入口
│  ├── search_space                # 搜索空间控制相关代码
│  ├── utils                  # ENAS核心代码
│  ├── scripts                 # 常用脚本目录，如直接调用主体代码脚本等。
│  ├── service                   # 提供主体代码的RESTFUL调用服务，服务框架如Flask、fastapi
│  ├── tests                     # 测试代码的目录
│  ├── ....                   
├── models                 # 模型文件存放目录
├── data                   # 数据文件存储目录
├── tests                     # 测试代码的目录
├── .gitignore                # git忽略追踪文件
├── LICENSE                   # 开源许可证明
├── requirements.txt          # 依赖文件
├── README.md                 # read me 文件
```

