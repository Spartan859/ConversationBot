"""
对话模型模块
"""

from .thu_agent import ThuAssistantAgent
from .general_agent import GeneralAgent
from .router import DialogueRouter

__all__ = ['ThuAssistantAgent', 'GeneralAgent', 'DialogueRouter']
