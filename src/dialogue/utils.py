"""
工具函数模块
提供文本处理、格式转换等通用功能
"""

import re


def markdown_to_speech_text(text: str) -> str:
    """
    将 Markdown 格式文本转换为适合 TTS 朗读的纯文本
    
    处理规则：
    - 去除 Markdown 标记（**加粗**、*斜体*、#标题、列表等）
    - 保留段落换行（便于停顿）
    - 处理链接格式 [文本](URL) → 文本
    - 去除代码块标记
    - 保留中文标点和自然语气
    
    Args:
        text: Markdown 格式文本
        
    Returns:
        适合朗读的纯文本
    
    Examples:
        >>> markdown_to_speech_text("**重要**：请提前办理")
        '重要：请提前办理'
        
        >>> markdown_to_speech_text("详见[学生手册](http://xxx)")
        '详见学生手册'
    """
    if not text:
        return ""
    
    # 0. 处理转义的换行符（\n 字面量）
    text = text.replace('\\n', '\n')
    
    # 1. 处理代码块（去除```标记）
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # 2. 处理标题标记（# ## ###）
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # 3. 处理加粗和斜体
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  # ***加粗斜体***
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)      # **加粗**
    text = re.sub(r'\*(.+?)\*', r'\1', text)          # *斜体*
    text = re.sub(r'__(.+?)__', r'\1', text)          # __加粗__
    text = re.sub(r'_(.+?)_', r'\1', text)            # _斜体_
    
    # 4. 处理链接 [文本](URL)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 5. 处理列表标记
    text = re.sub(r'^\s*[\-\*\+]\s+', '', text, flags=re.MULTILINE)  # 无序列表
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)     # 有序列表
    
    # 6. 处理引用标记
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
    
    # 7. 处理水平线
    text = re.sub(r'^[\-\*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # 8. 处理表格（简单去除|分隔符）
    text = re.sub(r'\|', ' ', text)
    
    # 9. 处理HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 10. 处理多余的空行（保留段落换行）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 11. 去除行首行尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # 12. 去除多余空格（保留中文之间的必要空格）
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 13. 最终清理
    text = text.strip()
    
    return text


def extract_xml_tag(text: str, tag: str) -> str:
    """
    从文本中提取指定 XML 标签的内容
    
    Args:
        text: 包含 XML 标签的文本
        tag: 标签名（不含尖括号）
        
    Returns:
        标签内容，如未找到返回空字符串
        
    Examples:
        >>> extract_xml_tag("<answer>你好</answer>", "answer")
        '你好'
    """
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def is_xml_tag_present(text: str, tag: str, value: str = None) -> bool:
    """
    检查 XML 标签是否存在，可选检查标签值
    
    Args:
        text: 待检查文本
        tag: 标签名
        value: 可选的标签值（如 "true"）
        
    Returns:
        是否存在匹配的标签
        
    Examples:
        >>> is_xml_tag_present("<need_agent>true</need_agent>", "need_agent", "true")
        True
    """
    if value:
        pattern = f'<{tag}>{value}</{tag}>'
    else:
        pattern = f'<{tag}>.*?</{tag}>'
    
    return bool(re.search(pattern, text, re.DOTALL))
