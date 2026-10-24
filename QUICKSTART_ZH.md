# 中文快速开始

安装好 Python 3.10+ 后，在项目根目录运行，无需安装依赖、下载模型或填写密钥：

```powershell
python -B -m journal_prefs assess --task examples/assessment_task.synthetic.json --request examples/assessment_request.synthetic.json --out my_first_demo
python -B -m unittest discover -s tests -v
```

演示使用虚构稿件、虚构评审和确定性 mock；分数只用于检查程序流程，不是实际效果。再次运行请换一个输出目录名，程序不会覆盖旧结果。

三个接口分别管理：模型服务及参数、偏好/评分算法、审稿标准及输出模板。可从 [数据格式](docs/DATA_SCHEMA.md)、[合成历史样本](examples/history.synthetic.jsonl)、[自定义算法](examples/custom_preferences.py)、[提示文件](docs/PROMPT_INTERFACE.md) 开始。

默认离线模式不联网；本地模式连接用户已经启动的回环地址兼容服务，不代为安装模型。云端配置默认禁用，需要明确设置端点、模型和传输许可，并可能产生服务费用。已有 HTTP 测试使用 mock，不能保证任意真实服务都兼容。不要把保密稿件放进 Git。

实际机构数据由机构自行授权和管理；样本须区分稿件版本、评审及评分时间，训练、验证、测试材料分开。完整历史格式和时间切分见 [技术说明](docs/TECHNICAL_GUIDE.md)。本仓库不附带真实论文、评审、评分缓存或模型输出，不保证审稿质量或录用概率。

真实规则条件对照仍未完成；没有基础模型微调、公开预检门户或生产部署。[当前状态](docs/EXPERIMENTS.md) · [English instructions](RUNNING.md)
