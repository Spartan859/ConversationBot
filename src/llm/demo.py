
import json
import os
import requests

from volcengine.auth.SignerV4 import SignerV4
from volcengine.base.Request import Request
from volcengine.Credentials import Credentials

collection_name = "thu"
project_name = "default"
query = "新生因故不能按期入学应如何处理？"
image_query = None
ak = os.getenv("THU_AGENT_AK", "")
sk = os.getenv("THU_AGENT_SK", "")
g_knowledge_base_domain = "api-knowledgebase.mlp.cn-beijing.volces.com"
account_id = os.getenv("THU_AGENT_ACCOUNT_ID", "")

base_prompt = """# 任务
你是一位清华大学的辅导员，你的首要任务是帮学生一步步解决他提出的问题。

你需要保证输出的回复为口语，文风需要适合直接说出来。

你需要根据「参考资料」来回答接下来的「用户问题」，这些信息在 <context></context> XML tags 之内。

 # 任务执行
现在请你根据提供的参考资料，回答用户的问题，你的回答需要准确和完整。

<context>
  {}
</context>"""

def prepare_request(method, path, params=None, data=None, doseq=0):
    if params:
        for key in params:
            if (
                    isinstance(params[key], int)
                    or isinstance(params[key], float)
                    or isinstance(params[key], bool)
            ):
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
        "Host": g_knowledge_base_domain,
        "V-Account-Id": account_id,
    }
    r.set_headers(headers)
    if params:
        r.set_query(params)
    r.set_host(g_knowledge_base_domain)
    r.set_path(path)
    if data is not None:
        r.set_body(json.dumps(data))

    # 生成签名
    credentials = Credentials(ak, sk, "air", "cn-north-1")
    SignerV4.sign(r, credentials)
    return r


def search_knowledge():
    method = "POST"
    path = "/api/knowledge/collection/search_knowledge"
    request_params = {
    "project": project_name,
    "name": collection_name,
    "query": query,
    "image_query": image_query,
    "limit": 10,
    "pre_processing": {
        "need_instruction": True,
        "return_token_usage": True,
        "messages": [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": ""
            }
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

    info_req = prepare_request(method=method, path=path, data=request_params)
    rsp = requests.request(
        method=info_req.method,
        url="http://{}{}".format(g_knowledge_base_domain, info_req.path),
        headers=info_req.headers,
        data=info_req.body
    )
    # print("search res = {}".format(rsp.text))
    return rsp.text

def chat_completion(message, stream=False, return_token_usage=True, temperature=0.7, max_tokens=4096):
    method = "POST"
    path = "/api/knowledge/chat/completions"
    request_params = {
        "messages": message,
        "stream": True,
        "return_token_usage": True,
        "model": "Deepseek-v3-1",
        "max_tokens": 32768,
        "temperature": 1,
        "model_version": "250821",
        "thinking": {
             "type":"enabled"
        }
    }

    info_req = prepare_request(method=method, path=path, data=request_params)
    rsp = requests.request(
        method=info_req.method,
        url="http://{}{}".format(g_knowledge_base_domain, info_req.path),
        headers=info_req.headers,
        data=info_req.body
    )
    rsp.encoding = "utf-8"
    print("chat completion res = {}".format(rsp.text))

def is_vision_model(model_name,model_version):
    if model_name is None:
        return False
    return "vision" in model_name or "m" in model_version
def get_content_for_prompt(point: dict) -> str:
    content = point["content"]
    original_question = point.get("original_question")
    if original_question:
        # faq 召回场景，content 只包含答案，需要把原问题也拼上
        return "当询问到相似问题时，请参考对应答案进行回答：问题：“{question}”。答案：“{answer}”".format(
                question=original_question, answer=content)
    return content

def generate_prompt(rsp_txt):
    rsp = json.loads(rsp_txt)
    if rsp["code"] != 0:
        return "", []
    prompt = ""
    rsp_data = rsp["data"]
    points = rsp_data["result_list"]
    using_vlm = is_vision_model("Deepseek-v3-1","250821")
    content = []

    for point in points:
        doc_text_part = ""
        # 先拼接系统字段
        doc_info = point["doc_info"]
        for system_field in ["point_id","doc_name","title","chunk_title","content"] : 
            if system_field == 'doc_name' or system_field == 'title':
                if system_field in doc_info:
                    doc_text_part += f"{system_field}: {doc_info[system_field]}\n"
            else:
                if system_field in point:
                    if system_field == "content":
                        doc_text_part += f"content: {get_content_for_prompt(point)}\n"
                    elif system_field == "point_id":
                        doc_text_part += f"point_id: \"{point['point_id']}\""
                    else:
                        doc_text_part += f"{system_field}: {point[system_field]}\n"
        if "table_chunk_fields" in point:
            table_chunk_fields = point["table_chunk_fields"]
            for self_field in [] : 
                # 使用 next() 从 table_chunk_fields 中找到第一个符合条件的项目
                find_one = next((item for item in table_chunk_fields if item["field_name"] == self_field), None)
                if find_one:
                    doc_text_part += f"{self_field}: {find_one['field_value']}\n"


        # 提取图片链接
        image_link = None
        if using_vlm and "chunk_attachment" in point:
            image_link = point["chunk_attachment"][0]["link"]
            if image_link:
                prompt += f"图片: \n"
        content.append({
            'type': 'text',
            'text': doc_text_part
        })
        if image_link:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': image_link
                }
            })
        prompt += f"{doc_text_part}\n"

    if using_vlm:
        content_pre_sub = base_prompt.split('{}')
        content_pre = {'type': 'text', 'text': content_pre_sub[0]}
        content_sub = {'type': 'text', 'text': content_pre_sub[1]}
        return [content_pre] + content + [content_sub]
    else:
        return base_prompt.format(prompt)


def search_knowledge_and_chat_completion():
    # 1.执行search_knowledge
    rsp_txt = search_knowledge()
    # 2.生成prompt
    prompt = generate_prompt(rsp_txt)
    # todo:用户需要本地缓存对话信息，并按照顺序依次加入到messages中
    # 3.拼接message对话, 问题对应role为user，系统对应role为system, 答案对应role为assistant, 内容对应content
    using_vlm = is_vision_model("Deepseek-v3-1", "250821")
    user_content = None
    if image_query:
        user_content = ([{"type": "image_url", "image_url": {"url": image_query}}] +
                      ([{"type": "text", "text": query}] if query else []))
    else:
        user_content = query
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    # 4.调用chat_completion
    chat_completion(messages)

if __name__ == "__main__":
    search_knowledge_and_chat_completion()
