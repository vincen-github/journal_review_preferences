# Journal Review Preferences

[English](README.md) | [简体中文](README.zh-CN.md)

**让每一本期刊，都能拥有理解自身评价标准的 AI 审稿助手。**

本项目致力于将机构积累的历史稿件、评审意见与录用决定，转化为可定制、可检验、可私有部署的审稿能力。机构掌握标准、数据与模型选择；项目免费开源。

> “prices, like everything else, move along the line of least resistance.”[1]

AI 降低了稿件生产成本，专家注意力却依然稀缺。通用模型能提供分析帮助，但不会自动理解一本期刊的标准。这里的阻力是各方投入的时间、资源与改变流程的代价，推动力则来自各自的收益。当出于自身利益作出的选择指向同一路径，改变便更容易发生：**投稿前反馈 → 编辑初筛 → 人工核验**。

- **作者：** 更早发现薄弱环节、判断期刊匹配，减少投错期刊与拒稿转投；未来可由机构提供可选的投稿前预检入口。
- **审稿人：** 减少无关阅读，把注意力留给需要深入判断的研究。
- **编辑：** 降低初筛与协调负担，更快定位分歧，同时保留有价值的稿件。
- **机构：** 扩展处理能力，维护质量、声誉和独特标准，保留数据与决策控制。

## 小接口，逐步采用

- 自定义评价维度与输出：文字评审、接收/修改/拒绝等分类、可选数字评分及其他字段。
- 由机构自行选择模型与偏好算法，分开管理训练、验证和测试材料。
- 替换本地或云端模型；本地服务地址、模型 ID、生成参数与结构化输出均可配置，首版无需微调基础模型。

离线 CLI 与可替换接口已可运行，真实规则条件对照尚未完成。[实现与实验状态](docs/EXPERIMENTS.md)。

评分预测已有 [PeerRead](https://github.com/allenai/PeerRead/tree/master/code) 等工作；本项目聚焦可配置的机构接口，效果增益仍待验证。

使用者也可能迎合模型偏好、寻找 shortcut，而非改善研究本身，因此设计上重视人工监督、正文证据与稳健性检查。

机构自行决定如何从历史材料形成评价标准；框架不指定必须采用的对齐算法。[通用评价](docs/ASSESSMENT_INTERFACE.md)、[算法示例](examples/custom_preferences.py)与[提示文件](docs/PROMPT_INTERFACE.md)相互独立，ICLR 评分只是一个例子。

## 本地体验

有 Python 3.10+ 时，在项目目录运行：

```powershell
python -B -m journal_prefs assess --task examples/assessment_task.synthetic.json --request examples/assessment_request.synthetic.json --out my_first_demo
```

[快速开始](QUICKSTART_ZH.md) · [技术说明](docs/TECHNICAL_GUIDE.md) · [数据试点](docs/PUBLIC_DATA_PILOT.md) · [验证记录](TEST_RESULTS.md) · [完整项目故事](PROJECT_RATIONALE_ZH.md)

代码采用 [MIT 许可](LICENSE)，[上游贡献与许可](THIRD_PARTY_NOTICES.md) 单独记录。dot 协助概念梳理、研究综合、协调与写作；Codex 代理完成实现及独立代码审查。[详细贡献](docs/TECHNICAL_GUIDE.md#project-contributions-and-ai-assistance)。

欢迎使用、改进和扩展本项目。如果其思路、设计或实现启发了您的工作，恳请注明出处，并链接至[本项目](README.zh-CN.md)，帮助读者追溯贡献。这是自愿致谢请求，不增加许可限制。

[1] Lefèvre, Edwin. *Reminiscences of a Stock Operator*. New York: George H. Doran Company, 1923. 本书为受 Jesse Livermore 经历启发的小说化叙述。[馆藏记录](https://catalog.hathitrust.org/Record/006583843)。

Wensen Ma
