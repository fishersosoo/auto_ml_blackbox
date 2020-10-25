# Sample
本项目是Python项目结构化的模板，目录结构仅供参考，根据项目实际情况进行调整。

## 目录结构
```
├── config                    # 配置文件目录
├── docker                    # 构建docker镜像的目录
├── docs                      # 放置项目文档的目录，如接口api文档、需求文档等
├── sample                    # 项目主体代码目录，该目录下的子目录仅供参考，根据实际情况调整
│  ├── __init__.py            # 主体代码的入口
│  ├── modules                # 可复用模块目录
│  ├── models                 # 模型文件存放目录
│  ├── data                   # 数据文件存储目录
│  ├── utils                  # 常用工具函数目录
│  ├── ....                   
├── scripts                   # 常用脚本目录，如直接调用主体代码脚本等。
├── service                   # 提供主体代码的RESTFUL调用服务，服务框架如Flask、fastapi
├── tests                     # 测试代码的目录
├── .gitignore                # git忽略追踪文件
├── LICENSE                   # 开源许可证明
├── requirements.txt          # 依赖文件
├── README.md                 # read me 文件
```