# 安全策略

## 报告安全漏洞

如果您发现了安全漏洞，请通过以下方式报告：

### 1. 私密报告（推荐）
- **邮箱**：huzekun123hzk-ship-it@github.com

### 2. GitHub 安全建议
- 使用 [GitHub Security Advisories](https://github.com/huzekun123hzk-ship-it/cs-vision-homework/security/advisories/new)
- 选择 "Report a vulnerability" 选项

### 3. 公开讨论
- 对于非敏感问题，可以在 [Issues](https://github.com/huzekun123hzk-ship-it/cs-vision-homework/issues) 中讨论

## 报告内容

请包含以下信息：

- 漏洞的详细描述
- 重现步骤
- 潜在影响评估
- 建议的修复方案（如果有）
- 您的联系信息

## 安全最佳实践

### 对于用户

1. **环境隔离**
   ```bash
   # 使用虚拟环境
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **依赖管理**
   ```bash
   # 定期更新依赖
   pip install --upgrade -r requirements.txt
   ```

3. **数据安全**
   - 不要将敏感数据提交到代码仓库
   - 使用 `.env` 文件管理配置
   - 定期清理临时文件

4. **代码审查**
   - 在运行实验前检查代码
   - 使用可信的数据源
   - 验证输入参数

### 对于贡献者

1. **代码安全**
   - 避免硬编码敏感信息
   - 使用安全的随机数生成
   - 验证所有输入参数

2. **依赖安全**
   - 定期检查依赖漏洞
   - 使用 `pip-audit` 扫描安全漏洞
   - 及时更新有漏洞的依赖

3. **提交安全**
   - 不要在提交中包含敏感信息
   - 使用 `.gitignore` 排除敏感文件
   - 定期检查提交历史

## 已知安全问题

目前没有已知的安全问题。

## 免责声明

本项目主要用于教学目的，请在使用前：

1. 仔细阅读代码和文档
2. 在安全环境中测试
3. 根据实际需求调整配置
4. 承担使用风险

## 联系方式

- **GitHub**：[huzekun123hzk-ship-it](https://github.com/huzekun123hzk-ship-it)

---

**注意**：请不要在公开的 Issues 或 Pull Requests 中讨论安全漏洞。请使用上述私密渠道报告。
