# 微信聊天记录导出指南

## 推荐工具：WeChatMsg（最简单）

**GitHub**: https://github.com/LC044/WeChatMsg

### 步骤

1. **确保微信在 PC 端登录过**（需要先在电脑上登录一次微信，让本地数据库生成）

2. **下载 WeChatMsg**
   - 前往 GitHub Releases 页面下载最新版 `.exe`
   - 或者 `pip install wechatmsg` 后运行 `python -m wechatmsg`

3. **运行工具，解密数据库**
   - 打开 WeChatMsg，它会自动定位微信的本地数据库
   - 点击"获取密钥"按钮，输入微信登录密码（用于解密）

4. **选择聊天对象，导出**
   - 在联系人列表中找到你想导出的聊天对象
   - 选择导出格式：**JSON**（推荐）
   - 点击导出，保存到 `data/raw/` 目录下

5. **表情包导出**（可选，Phase 5 需要）
   - WeChatMsg 支持同时导出自定义表情包图片
   - 导出后放到 `data/processed/stickers/` 目录

---

## 备选工具：PyWxDump（技术用户）

**GitHub**: https://github.com/xaoyaoo/PyWxDump

适合更高级的定制需求，支持导出更多数据类型。

```bash
pip install pywxdump
pywxdump ui  # 启动 Web UI
```

---

## 导出后

将导出的 JSON 或 CSV 文件放入 `data/raw/` 目录，然后运行：

```bash
# 自动检测昵称
python scripts/preprocess.py --input data/raw/ --output data/processed

# 手动指定昵称（推荐，更准确）
python scripts/preprocess.py \
  --input data/raw/chat_export.json \
  --output data/processed \
  --her "她的微信昵称" \
  --you "你的微信昵称"
```

---

## 常见问题

**Q: 找不到微信数据库在哪里？**
A: 通常在 `C:\Users\用户名\Documents\WeChat Files\`，WeChatMsg 会自动定位。

**Q: 解密失败怎么办？**
A: 确保微信是在当前电脑上登录的，且当前微信版本支持。WeChatMsg 的 GitHub Issues 里有常见解决方案。

**Q: 导出的数据安全吗？**
A: 所有数据均在本地处理，不会上传。API 调用只会发送 prompt 内容（不包含完整聊天记录）。
