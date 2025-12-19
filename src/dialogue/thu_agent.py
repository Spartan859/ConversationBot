"""
清华本科学习助手 Agent
基于火山引擎知识库 API 实现 RAG 增强对话
"""

import json
import requests
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

from volcengine.auth.SignerV4 import SignerV4
from volcengine.base.Request import Request
from volcengine.Credentials import Credentials

from .utils import markdown_to_speech_text


class ThuAssistantAgent:
    """清华大学本科学习助手 Agent"""
    
    # 默认的 System Prompt
    DEFAULT_BASE_PROMPT = """# 任务
你是一位清华大学的辅导员，你的首要任务是帮学生一步步解决他提出的问题。

你需要保证输出的回复为口语，文风需要适合直接说出来。

你需要根据「参考资料」来回答接下来的「用户问题」，这些信息在 <context></context> XML tags 之内。

# 任务执行
现在请你根据提供的参考资料，回答用户的问题，你的回答需要准确和完整。

<context>
  {}
</context>"""
    
    def __init__(
        self,
        ak: str,
        sk: str,
        account_id: str,
        collection_name: str = "thu",
        project_name: str = "default",
        model: str = "Deepseek-v3-1",
        model_version: str = "250821",
        base_prompt: Optional[str] = None,
        domain: str = "api-knowledgebase.mlp.cn-beijing.volces.com"
    ):
        """
        初始化清华助手 Agent
        
        Args:
            ak: 火山引擎 Access Key
            sk: 火山引擎 Secret Key
            account_id: 账户 ID
            collection_name: 知识库名称
            project_name: 项目名称
            model: 模型名称
            model_version: 模型版本
            base_prompt: 自定义 System Prompt（可选）
            domain: API 域名
        """
        self.ak = ak
        self.sk = sk
        self.account_id = account_id
        self.collection_name = collection_name
        self.project_name = project_name
        self.model = model
        self.model_version = model_version
        self.domain = domain
        
        # 使用自定义 prompt 或默认 prompt
        self.base_prompt = base_prompt or self.DEFAULT_BASE_PROMPT
    
    def _prepare_request(
        self, 
        method: str, 
        path: str, 
        params: Optional[Dict] = None, 
        data: Optional[Dict] = None, 
        doseq: int = 0
    ) -> Request:
        """准备 HTTP 请求"""
        if params:
            for key in params:
                if isinstance(params[key], (int, float, bool)):
                    params[key] = str(params[key])
                elif isinstance(params[key], list):
                    if not doseq:
                        params[key] = ",".join(params[key])
        
        r = Request()
        r.set_shema("http")
        r.set_method(method)
        r.set_connection_timeout(10)
        r.set_socket_timeout(10)
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "Host": self.domain,
            "V-Account-Id": self.account_id,
        }
        r.set_headers(headers)
        
        if params:
            r.set_query(params)
        
        r.set_host(self.domain)
        r.set_path(path)
        
        if data is not None:
            r.set_body(json.dumps(data))
        
        # 生成签名
        credentials = Credentials(self.ak, self.sk, "air", "cn-north-1")
        SignerV4.sign(r, credentials)
        
        return r
    
    def search_knowledge(
        self, 
        query: str, 
        image_query: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """
        从知识库检索相关文档
        
        Args:
            query: 查询文本
            image_query: 图片查询（可选）
            limit: 返回结果数量
            
        Returns:
            检索结果的 JSON 字符串
        """
        method = "POST"
        path = "/api/knowledge/collection/search_knowledge"
        
        request_params = {
            "project": self.project_name,
            "name": self.collection_name,
            "query": query,
            "image_query": image_query,
            "limit": limit,
            "pre_processing": {
                "need_instruction": True,
                "return_token_usage": True,
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": ""}
                ],
                "rewrite": False
            },
            "post_processing": {
                "get_attachment_link": True,
                "rerank_only_chunk": False,
                "rerank_switch": False,
                "chunk_group": True,
                "rerank_model": "doubao-seed-rerank",
                "enable_rerank_threshold": False,
                "retrieve_count": 25,
                "chunk_diffusion_count": 0
            },
            "dense_weight": 0.5
        }
        
        info_req = self._prepare_request(method=method, path=path, data=request_params)
        rsp = requests.request(
            method=info_req.method,
            url=f"http://{self.domain}{info_req.path}",
            headers=info_req.headers,
            data=info_req.body
        )
        
        return rsp.text
    
    def _is_vision_model(self) -> bool:
        """判断是否为视觉模型"""
        return "vision" in self.model.lower() or "m" in self.model_version
    
    def _get_content_for_prompt(self, point: Dict) -> str:
        """获取知识点内容（处理 FAQ 场景）"""
        content = point["content"]
        original_question = point.get("original_question")
        
        if original_question:
            return f'当询问到相似问题时，请参考对应答案进行回答：问题："{original_question}"。答案："{content}"'
        
        return content
    
    def generate_prompt(self, search_result: str) -> Any:
        """
        根据检索结果生成 Prompt
        
        Args:
            search_result: search_knowledge 返回的 JSON 字符串
            
        Returns:
            格式化后的 Prompt（文本或多模态内容列表）
        """
        rsp = json.loads(search_result)
        
        if rsp["code"] != 0:
            return ""
        
        prompt = ""
        rsp_data = rsp["data"]
        points = rsp_data["result_list"]
        using_vlm = self._is_vision_model()
        content = []
        
        for point in points:
            doc_text_part = ""
            doc_info = point["doc_info"]
            
            # 拼接系统字段
            for system_field in ["point_id", "doc_name", "title", "chunk_title", "content"]:
                if system_field in ['doc_name', 'title']:
                    if system_field in doc_info:
                        doc_text_part += f"{system_field}: {doc_info[system_field]}\n"
                else:
                    if system_field in point:
                        if system_field == "content":
                            doc_text_part += f"content: {self._get_content_for_prompt(point)}\n"
                        elif system_field == "point_id":
                            doc_text_part += f"point_id: \"{point['point_id']}\"\n"
                        else:
                            doc_text_part += f"{system_field}: {point[system_field]}\n"
            
            # 处理表格字段（如有需要）
            if "table_chunk_fields" in point:
                table_chunk_fields = point["table_chunk_fields"]
                for self_field in []:  # 可根据需要添加字段
                    find_one = next(
                        (item for item in table_chunk_fields if item["field_name"] == self_field), 
                        None
                    )
                    if find_one:
                        doc_text_part += f"{self_field}: {find_one['field_value']}\n"
            
            # 提取图片链接（多模态）
            image_link = None
            if using_vlm and "chunk_attachment" in point:
                image_link = point["chunk_attachment"][0]["link"]
                if image_link:
                    prompt += "图片: \n"
            
            content.append({
                'type': 'text',
                'text': doc_text_part
            })
            
            if image_link:
                content.append({
                    'type': 'image_url',
                    'image_url': {'url': image_link}
                })
            
            prompt += f"{doc_text_part}\n"
        
        # 根据是否为视觉模型返回不同格式
        if using_vlm:
            content_pre_sub = self.base_prompt.split('{}')
            content_pre = {'type': 'text', 'text': content_pre_sub[0]}
            content_sub = {'type': 'text', 'text': content_pre_sub[1]}
            return [content_pre] + content + [content_sub]
        else:
            return self.base_prompt.format(prompt)
    
    def chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        stream: bool = False,
        max_tokens: int = 32768,
        temperature: float = 1.0,
        enable_thinking: bool = True
    ) -> str:
        """
        调用大模型生成回答
        
        Args:
            messages: 对话消息列表
            stream: 是否流式输出
            max_tokens: 最大 token 数
            temperature: 温度参数
            enable_thinking: 是否启用思考模式
            
        Returns:
            模型生成的回答
        """
        method = "POST"
        path = "/api/knowledge/chat/completions"
        
        request_params = {
            "messages": messages,
            "stream": stream,
            "return_token_usage": True,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model_version": self.model_version,
        }
        
        if enable_thinking:
            request_params["thinking"] = {"type": "enabled"}
        
        info_req = self._prepare_request(method=method, path=path, data=request_params)
        rsp = requests.request(
            method=info_req.method,
            url=f"http://{self.domain}{info_req.path}",
            headers=info_req.headers,
            data=info_req.body
        )
        rsp.encoding = "utf-8"
        
        return rsp.text
    
    def query(
        self, 
        user_query: str, 
        image_query: Optional[str] = None,
        max_tokens: int = 32768,
        temperature: float = 1.0,
        post_process: bool = False
    ) -> str:
        """
        完整的 RAG 查询流程
        
        Args:
            user_query: 用户问题
            image_query: 图片查询（可选）
            max_tokens: 最大 token 数
            temperature: 温度参数
            post_process: 是否后处理为纯文本（去除 Markdown 标记，适合 TTS）
            
        Returns:
            Agent 的回答（Markdown 格式或纯文本）
        """
        # 1. 知识检索
        search_result = self.search_knowledge(user_query, image_query)
        
        # 2. 生成 Prompt
        system_prompt = self.generate_prompt(search_result)
        
        # 3. 构建消息
        using_vlm = self._is_vision_model()
        
        if image_query:
            user_content = [
                {"type": "image_url", "image_url": {"url": image_query}}
            ]
            if user_query:
                user_content.append({"type": "text", "text": user_query})
        else:
            user_content = user_query
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # 4. 调用大模型
        response = self.chat_completion(
            messages, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        # 5. 后处理（如果需要）
        if post_process:
            # 提取实际回答内容（处理流式响应）
            try:
                response_data = json.loads(response)
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0].get("message", {}).get("content", "")
                    # 转换为纯文本
                    content = markdown_to_speech_text(content)
                    return content
            except json.JSONDecodeError:
                pass
            
            # 如果解析失败，直接对原始文本进行处理
            return markdown_to_speech_text(response)
        
        return response
    
    def set_model(self, model: str, model_version: str):
        """切换模型"""
        self.model = model
        self.model_version = model_version
    
    def set_base_prompt(self, base_prompt: str):
        """更新 System Prompt"""
        self.base_prompt = base_prompt


    # 无 __main__，请使用 demo_agents.py 运行示例
