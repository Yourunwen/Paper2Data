#!/usr/bin/env python3
"""
SGLang LLM Client - 本地 LLM 客户端封装

与 sglang_server.py 配合使用，提供简洁的本地 LLM 调用接口。
实现 data_meta_extract.py 中的元数据提取功能（仅针对本地 LLM）。

使用方法：
    from sglang_LLM import SGLangClient
    
    client = SGLangClient(base_url="http://127.0.0.1:30000", model_name="Qwen/Qwen3-4B-Instruct-2507")
    response = client.chat(prompt="你好", system_prompt="You are a helpful assistant.")
    print(response)
"""

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv

# 尝试导入 openai 客户端（推荐方式）
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # 大规模批处理时可静默 INFO/WARN 输出：export SGLANG_SILENT=1
    if os.getenv("SGLANG_SILENT") != "1": print("[WARN] openai 库未安装，将使用 requests。建议安装: pip install openai")


def _log_info(msg: str) -> None:
    """可通过环境变量 SGLANG_SILENT=1 关闭 INFO 日志。"""
    if os.getenv("SGLANG_SILENT") == "1":
        return
    print(msg)


class SGLangClient:
    """SGLang 本地 LLM 客户端（OpenAI 兼容）"""
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 8196,
        timeout: int = 600,  # 增加到10分钟（transformers后端可能较慢）
    ):
        """
        初始化 SGLang 客户端
        
        Args:
            base_url: 服务器地址（默认：http://127.0.0.1:30000）
            model_name: 模型名称（默认：Qwen/Qwen3-4B-Instruct-2507）
            api_key: API 密钥（本地服务器通常不需要，默认：EMPTY）
            temperature: 生成温度（默认：0.0，确定性生成）
            max_tokens: 最大生成 token 数（默认：2048）
            timeout: 请求超时时间（秒）（默认：300）
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        # 清理 API key：去除首尾空白和换行符，避免 HTTP header 格式错误
        self.api_key = (api_key.strip() if api_key else "EMPTY").replace('\n', '').replace('\r', '')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # 优先使用 openai 客户端
        if OPENAI_AVAILABLE:
            self.client = openai.Client(
                base_url=f"{self.base_url}/v1",
                api_key=self.api_key  # 使用清理后的 API key
            )
            self.use_openai = True
        else:
            self.client = None
            self.use_openai = False
    
    def chat(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        调用 LLM 进行对话
        
        Args:
            prompt: 用户输入的提示词
            system_prompt: 系统提示词（默认：You are a helpful AI assistant.）
            temperature: 生成温度（覆盖默认值）
            max_tokens: 最大生成 token 数（覆盖默认值）
        
        Returns:
            LLM 生成的文本
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.use_openai:
            # 使用 openai 客户端
            try:
                _log_info(f"[INFO] 发送请求到: {self.base_url}/v1/chat/completions")
                _log_info(f"[INFO] 超时时间: {self.timeout} 秒")
                _log_info(f"[INFO] 提示词长度: {len(prompt)} 字符")
                
                import time
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                )
                
                elapsed_time = time.time() - start_time
                _log_info(f"[INFO] 请求完成，耗时: {elapsed_time:.2f} 秒")
                
                content = response.choices[0].message.content
                _log_info(f"[INFO] 响应长度: {len(content)} 字符")
                
                return content
            except Exception as exc:
                print(f"[ERROR] OpenAI 客户端调用失败: {exc}")
                print(f"[HELP] 请检查：")
                print(f"[HELP]   1. 服务器是否正在运行？")
                print(f"[HELP]   2. 服务器日志中是否有错误信息？")
                print(f"[HELP]   3. 是否超时？（当前超时: {self.timeout} 秒）")
                raise
        else:
            # 使用 requests
            try:
                url = f"{self.base_url}/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {self.api_key}",
                }
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                _log_info(f"[INFO] 发送请求到: {url}")
                _log_info(f"[INFO] 超时时间: {self.timeout} 秒")
                _log_info(f"[INFO] 提示词长度: {len(prompt)} 字符")
                
                import time
                start_time = time.time()
                
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                
                elapsed_time = time.time() - start_time
                _log_info(f"[INFO] 请求完成，耗时: {elapsed_time:.2f} 秒")
                
                data = response.json()
                
                choices = data.get("choices")
                if not choices:
                    raise ValueError("LLM 返回结构异常")
                
                content = choices[0]["message"]["content"].strip()
                _log_info(f"[INFO] 响应长度: {len(content)} 字符")
                
                return content
            except requests.Timeout:
                print(f"[ERROR] 请求超时（{self.timeout} 秒）")
                print(f"[HELP] transformers 后端可能需要更长时间，建议：")
                print(f"[HELP]   1. 增加超时时间（当前: {self.timeout} 秒）")
                print(f"[HELP]   2. 使用 vLLM 或 SGLang（速度更快）")
                print(f"[HELP]   3. 减少 max_tokens（当前: {max_tokens}）")
                raise
            except requests.HTTPError as http_exc:
                print(f"[ERROR] HTTP 错误: {http_exc}")
                print(f"[ERROR] 响应内容: {response.text if 'response' in locals() else 'N/A'}")
                raise
            except Exception as exc:
                print(f"[ERROR] Requests 调用失败: {exc}")
                print(f"[HELP] 请检查：")
                print(f"[HELP]   1. 服务器是否正在运行？")
                print(f"[HELP]   2. 服务器日志中是否有错误信息？")
                raise
    
    def test_connection(self) -> bool:
        """
        测试与服务器的连接
        
        Returns:
            连接是否成功
        """
        try:
            # 尝试获取模型列表
            url = f"{self.base_url}/v1/models"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            _log_info(f"[INFO] 连接成功: {self.base_url}")
            return True
        except Exception as exc:
            print(f"[ERROR] 连接失败: {exc}")
            return False


# ==================== 元数据提取功能（与 data_meta_extract.py 兼容） ====================

def _extract_think_content(text: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    提取并移除 <think>...</think> 中的思考内容。
    
    Args:
        text: 原始文本
        
    Returns:
        (剥离后的文本, 思考内容) 如果找到思考内容，否则返回 (原始文本, None)
    """
    if not text:
        return "", None
    
    think_content = None
    original_text = text
    
    # 提取标准闭合标签中的内容（支持 <think> 标签）
    think_pattern = r"(?is)<think>(.*?)</think>"
    matches = re.findall(think_pattern, text)
    if matches:
        # 合并所有匹配的思考内容
        think_content = "\n\n".join(match.strip() for match in matches if match.strip())
        # 移除思考标签，但保留其他内容
        stripped = re.sub(think_pattern, "", text)
    else:
        # 检查是否有未闭合的标签
        unclosed_pattern = r"(?is)<think>(.*)$"
        match = re.search(unclosed_pattern, text)
        if match:
            think_content = match.group(1).strip()
            stripped = re.sub(unclosed_pattern, "", text)
        else:
            stripped = text
    
    # 如果提取了思考内容，确保剩余文本不为空
    stripped = stripped.strip()
    if think_content and not stripped:
        # 如果提取后内容为空，可能是格式问题，返回原始文本
        print(f"[WARN] 提取思考内容后剩余文本为空，使用原始文本")
        return original_text, think_content
    
    return stripped, think_content if think_content else None


def _extract_json_from_text(text: str) -> Optional[str]:
    """从文本中提取 JSON 内容（与 data_meta_extract.py 一致）"""
    if not text:
        return None
    
    text = text.strip()
    
    # 1. 尝试直接解析（纯 JSON）
    try:
        json.loads(text)
        return text
    except:
        pass
    
    # 2. 尝试提取代码块中的 JSON
    # 匹配 ```json ... ``` 或 ``` ... ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                json.loads(match.strip())
                return match.strip()
            except:
                continue
    
    # 3. 尝试找到 JSON 数组或对象
    json_patterns = [
        r'(\[[\s\S]*\])',  # JSON 数组
        r'(\{[\s\S]*\})',  # JSON 对象
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                start_idx = text.find(match)
                if start_idx == -1:
                    continue
                
                remaining = text[start_idx:]
                bracket_count = 0
                brace_count = 0
                end_idx = -1
                
                for i, char in enumerate(remaining):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                    elif char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    
                    if (bracket_count == 0 and brace_count == 0) and (remaining[0] == '[' or remaining[0] == '{'):
                        end_idx = i + 1
                        break
                
                if end_idx > 0:
                    json_candidate = remaining[:end_idx]
                    try:
                        json.loads(json_candidate)
                        return json_candidate
                    except:
                        pass
            except:
                continue
    
    return None


def _normalize_llm_output(parsed_content, debug_content: Optional[str] = None) -> List[Dict]:
    """规范化和验证 LLM 返回的 JSON 内容（与 data_meta_extract.py 一致）"""
    def _cap_and_prioritize_ref_ids(ref_ids: List[str], max_n: int = 3) -> List[str]:
        """
        将 ref 规范化为“少量最相关 ID”，避免模型把整段 Reference IDs 串复制进来。
        优先级：Data Citation > Supplementary Data > ref-CRxx > 其它字符串
        """
        if not ref_ids:
            return []
        # 去重保持顺序
        seen = set()
        uniq: List[str] = []
        for x in ref_ids:
            if not isinstance(x, str):
                continue
            t = x.strip()
            if not t:
                continue
            if t not in seen:
                seen.add(t)
                uniq.append(t)

        def rank(x: str) -> int:
            xl = x.lower()
            if xl.startswith("data citation"):
                return 0
            if xl.startswith("supplementary data"):
                return 1
            if re.fullmatch(r"ref-CR\d+", x):
                return 2
            return 3

        # 稳定排序：先按优先级，再按原始顺序（Python sort 稳定）
        ordered = sorted(uniq, key=rank)
        return ordered[:max_n]

    if parsed_content is None:
        if debug_content:
            print(f"[WARN] LLM 返回 null（原始内容前200字符: {debug_content[:200] if len(debug_content) > 200 else debug_content}）")
        print(f"[WARN] LLM 返回 null，将使用空列表")
        return []
    elif isinstance(parsed_content, dict):
        print(f"[WARN] LLM 返回单个对象而不是数组，将包装成数组")
        return [parsed_content]
    elif isinstance(parsed_content, list):
        validated = []
        for idx, item in enumerate(parsed_content):
            if isinstance(item, dict):
                # 兼容 ref 字段：统一为数组（列表）形式
                # - None/null -> []
                # - string -> 尝试提取引用ID（ref-CRxx / Data Citation X / Supplementary Data X），否则放入单元素列表
                # - list -> 保留字符串元素
                ref_val = item.get("ref", [])
                if ref_val is None:
                    item["ref"] = []
                elif isinstance(ref_val, str):
                    ids = re.findall(r"ref-CR\d+", ref_val)
                    ids += re.findall(r"Data Citation\s*\d+", ref_val, flags=re.IGNORECASE)
                    ids += re.findall(r"Supplementary Data\s*\d+", ref_val, flags=re.IGNORECASE)
                    normalized = _cap_and_prioritize_ref_ids(ids, max_n=3)
                    item["ref"] = normalized if normalized else ([ref_val.strip()] if ref_val.strip() else [])
                elif isinstance(ref_val, list):
                    cleaned = []
                    seen = set()
                    for x in ref_val:
                        if not isinstance(x, str):
                            continue
                        x2 = x.strip()
                        if not x2:
                            continue
                        if x2 not in seen:
                            seen.add(x2)
                            cleaned.append(x2)
                    item["ref"] = _cap_and_prioritize_ref_ids(cleaned, max_n=3)
                else:
                    item["ref"] = []
                validated.append(item)
            elif item is None:
                print(f"[WARN] 数组中的第 {idx} 个元素是 null，已跳过")
            else:
                print(f"[WARN] 数组中的第 {idx} 个元素不是字典类型（{type(item)}），已跳过")
        return validated
    else:
        print(f"[WARN] LLM 返回了意外的类型: {type(parsed_content)}，将使用空列表")
        if debug_content:
            print(f"[DEBUG] 原始内容: {debug_content[:500]}")
        return []


def extract_metadata_with_sglang(
    abstract: str,
    data_availability: Union[str, Dict, None],
    methods_text: Optional[str],
    client: SGLangClient,
    max_attempts: int = 3,
    base_delay: int = 5,
    full_article_text: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str], Dict]:
    """
    使用 SGLang 客户端提取元数据（与 data_meta_extract.py 的 call_llm_for_metadata 兼容）
    
    Args:
        abstract: 文章摘要
        data_availability: 数据可用性声明
        methods_text: 方法部分文本
        client: SGLangClient 实例
        max_attempts: 最大重试次数（默认：3）
        base_delay: 基础延迟时间（秒）（默认：5）
        full_article_text: 当需要使用全文提示时传入（默认：None）
    
    Returns:
        (提取的数据集元数据列表, 思考内容, 解析失败信息) 
        思考内容可能为 None
        解析失败信息是一个字典，包含：
        - first_attempt_failed: 第一次尝试是否失败
        - all_attempts_failed: 所有重试是否都失败
        - error_message: 第一次失败时的错误信息
        - failed_attempts: 失败的尝试次数
    """
    # 构建系统提示词（与 data_meta_extract.py 一致，并强化“原始数据”定义）
    # 旧版提示语，仅参考：
    # system_prompt = (
    #     "You are a meticulous research assistant who extracts and synthesizes metadata from academic articles, focusing on dataset descriptions. "
    #     "Your task is to read the abstract, methods and Data Availability statement of the provided article, and extract every ORIGINAL dataset that is used as INPUT data for the research. "
    #     ...
    # )
    system_prompt = (
        "You are a meticulous research assistant who extracts and synthesizes metadata from academic articles, focusing on dataset descriptions. "
        "You will be given the ENTIRE article content (all sections except Meta_info). "
        "Carefully read the full article and extract every ORIGINAL dataset that is used as INPUT data for the research. "
        "ORIGINAL datasets include: (a) external databases, official statistics, published datasets, or open data portals that provide concrete data used in the study; "
        "(b) data collected directly by the authors and documented in the article (such as surveys, interviews, experiments, observations, measurements, sensor recordings, or fieldwork). "
        "Do NOT treat tables, figures, visualization outputs, or model results that only present analysis findings as separate datasets, unless the paper explicitly states that a new dataset or data product is being released for reuse. "
        "Do NOT treat purely theoretical constructs, indices, or models as datasets unless the paper clearly uses an underlying data file or database of their values as input. "
        "For each dataset, provide a clear summary of its content, purpose, and application based on the full article context. "
        "Ensure you understand the dataset's role in the study."
    )
    
    # 提取文本
    def _extract_text_from_section(value):
        if not value:
            return None
        if isinstance(value, dict):
            content = value.get("content", "")
            links = value.get("links") or value.get("Links") or {}
            text_parts = [content.strip()] if content else []
            if isinstance(links, dict) and links:
                link_lines = [f"{k}: {v}" for k, v in links.items()]
                text_parts.append("Links:\n" + "\n".join(link_lines))
            elif isinstance(links, list) and links:
                text_parts.append("Links:\n" + "\n".join(str(item) for item in links))
            merged = "\n".join(part for part in text_parts if part)
            return merged or None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None
    # print("MMMMMMM"*30)
    # print(methods_text)
    # print("MMMMMMM"*30)
    
    data_availability_text = _extract_text_from_section(data_availability) or "N/A"
    methods_text_processed = _extract_text_from_section(methods_text) or "N/A" 

    # 构建用户提示词（与 data_meta_extract.py 一致，并细化规则）
    section_blocks = [
    "Within the full article content provided (all sections except Meta_info), provide metadata for each ORIGINAL dataset mentioned. "
    "ORIGINAL datasets include external data products cited from other publications, public databases, government repositories, open data portals, benchmark datasets, platform logs, remote sensing products, or supplementary data released by other authors, "
    "as well as any data collected directly within this article (surveys, measurements, experiments, interviews, observations, sensor recordings, fieldwork, manual coding/annotation, web scraping, etc.). "

    "FOCUS: Prefer datasets used as INPUT data sources in the study workflow. "
    "If a dataset-like data source is plausibly used as input but the text is ambiguous, you MAY still include it, but set low confidence in the Evidence for the relevant fields. "
    "Do NOT treat analysis results, processed outputs, tables/figures that only show findings, visualization-only outputs, or model-generated data as datasets, unless the paper explicitly releases them as a reusable dataset/data product. "
    "Do NOT treat purely theoretical concepts, indices, or models as datasets unless a concrete data file/database is used as input. "

    "FIGURE/CAPTION NOTE: You may scan figure captions for dataset mentions. "
    "Only extract a dataset from a figure caption if it explicitly names a concrete data source; do not extract datasets from captions that only describe results or schematics. "

    "DE-DUPLICATION (lightweight): Output each dataset at most ONCE when it is clearly the same dataset by explicit alias/abbreviation statements. "
    "Do NOT deduplicate based on repository/platform/provider/domain similarity alone. If unsure, keep them separate. "

    "IMPORTANT: You must output ONLY valid JSON, nothing else. "
    "Output must be a JSON array (even if empty). Start directly with '[' and end with ']'. No markdown. "

    "Each dataset object must have exactly these keys:\n"
    '  - "Data_Name": A concise and clear name for the dataset.\n'
    '  - "Data_summary": A detailed description of what the dataset is (data type, source, and granularity) and how it is used in this study. '
    'Focus on describing the nature of the data itself rather than study findings or methodology. '
    'Do NOT restate time coverage, geographic coverage, sample sizes, access conditions, URLs, or citation identifiers here.\n'    '  - "Category": The category of the dataset. This must be one of the following options:\n'
    '    "Road networks and transportation infrastructure"\n'
    '    "Building footprints and land use maps"\n'
    '    "Points of Interest (POIs)"\n'
    '    "Administrative boundaries and zoning maps"\n'
    '    "Utility networks (electricity, water, communication)"\n'
    '    "Human mobility traces (GPS, transit card, ride-hailing)"\n'
    '    "Socioeconomic activities (consumption, employment, commerce)"\n'
    '    "Social media interactions and online behavior"\n'
    '    "Health and wellbeing data (hospitalization, surveys)"\n'
    '    "Population census and household surveys"\n'
    '    "Statistical yearbooks and socioeconomic indicators"\n'
    '    "Government reports and planning documents"\n'
    '    "Policy texts and regulatory frameworks"\n'
    '    "Satellite remote sensing imagery (optical, SAR, night lights)"\n'
    '    "Aerial and drone imagery"\n'
    '    "Ground-based sensors (air quality, temperature, noise)"\n'
    '    "Urban IoT devices (traffic, energy, water, environmental monitoring)"\n'
    '    "City-wide camera networks and meteorological stations"\n'
    '    "null"\n'
    '  - "Need_Author_Contact": true/false/null. true ONLY if explicitly available upon request/contact authors/restricted approval required; false if explicitly public/open or direct access provided; null if not mentioned.\n'
    '  - "InText_Citation_Numbers": array of integers extracted ONLY from numeric citations directly attached to the dataset mention (same sentence or immediately following citation). Max 3. If none, [].\n'

    '  - "Other_Information": Any additional relevant notes, such as data preprocessing, limitations, or contextual information that does not fit into the other fields.\n'
    '    IMPORTANT (evidence required): You MUST include a short evidence snippet and its location in the article inside this field.\n'
    '    Use this exact format (single string): "Evidence: <quote 1 sentence, <=200 chars>; Location: <Section title / Data Availability / Methods / etc.>; Confidence: <high|medium|low>".\n'
    '    JSON-safety rules for this field: Do NOT wrap Evidence in quotes, and do NOT include any double quotes (") or semicolons (;) inside Evidence or Location. '
    '    If the original sentence contains quotes or semicolons, remove them or replace " with \' and ; with ,.\n'
    '  - "Time_Coverage": temporal coverage. If missing, "N/A". Prefer formats like "YYYY-YYYY", "YYYY-MM to YYYY-MM", or "YYYY-MM-DD to YYYY-MM-DD".\n'
    '  - "Geographic_Coverage": geographic scope. If missing, "N/A". Prefer "Country > State/Province > City/Site" when possible.\n'
    '  - "URL": dataset access link/DOI/repository. If missing, "N/A".\n'
    '    Umbrella rule: You may create ONE separate object with Data_Name="Paper-level data deposit (from Data Availability)" ONLY if the text clearly states ALL data (or data supporting findings / all datasets used) are available at one URL X. Do NOT copy X into every dataset.\n'
    '  - "ref": array of explicit IDs only (0–3), e.g., ref-CRxx, Data Citation X, Supplementary Data X. Do NOT infer from [number] or author-year.\n'
    '  - "Evidence": field-level evidence for verification. This must be a JSON object with keys:\n'
    '      {"Data_Name": {...}, "Category": {...}, "Need_Author_Contact": {...}, "InText_Citation_Numbers": {...}, "Time_Coverage": {...}, "Geographic_Coverage": {...}, "URL": {...}, "ref": {...}}\n'
    '    Each value is ONE object (not an array): {"quote": <<=260 chars>, "location": <section>, "confidence": <high|medium|low>}.\n'
    '    quote must be copied verbatim from the article text, not paraphrased or inferred.\n'
    '    If a field is N/A/null/[] because not reported, you may set quote="not reported", confidence="low".\n'

]




 


#"URL": A valid link where the dataset can be accessed. If no valid link is provided, write "N/A".\n\n'
   
# - "URL": A valid access point or reference identifier associated with the dataset. Acceptable values include:
#     • Direct dataset download links or official project/data repository URLs (e.g., GitHub, Zenodo, Figshare, Kaggle, institutional repositories).
#     • Official websites of the organizations or agencies that publish or maintain the dataset, when these serve as the dataset’s primary access point.
#     • Application/request portals for restricted-access datasets (e.g., government data request pages, controlled-access systems).
#     • If the dataset is obtained through a referenced publication in the paper and no real URL is provided, use the reference identifier exactly as written (e.g., "ref-CR1", "ref-CR11") instead of a real link.
#     • If the paper itself is the only source of the dataset (e.g., data embedded in tables, figures, or supplementary materials), use the reference identifier for the paper (e.g., "ref-CR0") or its citation ID used in the document.
#   If no access link, no identifiable reference identifier, and no traceable source exists, write "N/A".

    if full_article_text:
        section_blocks.append(
            "You are provided with the full article content below (all sections except Meta_info). "
            "Use it entirely when identifying original datasets.\n"
            "Full article content (excluding Meta_info):\n"
            + full_article_text
        )
    else:
        section_blocks.append("Article abstract:\n" + (abstract or "N/A"))
        if methods_text_processed and methods_text_processed != "N/A":
            section_blocks.append("Methods section:\n" + methods_text_processed)
        section_blocks.append("Data Availability statement:\n" + data_availability_text)
    
    # 再次强调输出格式
    # section_blocks.append(
    #     "\n\nREMEMBER: Output ONLY a valid JSON array. Start with '[' and end with ']'. "
    #     "Do NOT include any explanations, comments, or natural language text. "
    #     "If no datasets are found, output an empty array: []"
    # )
    # Phase-1（候选生成器）策略：不再强制“先确定 N 且输出 exactly N”，以提升召回与跨模型一致性。
    # 第二阶段（LLM_check.py）会负责 DROP/MERGE 精炼。
    # section_blocks.append(
    #     "Phase-1 recall strategy: Try to be as complete as possible and avoid missing any plausible INPUT datasets. "
    #     "If you are unsure whether something counts as an input dataset, INCLUDE it as a candidate rather than omitting it, "
    #     "and indicate uncertainty by adding 'Confidence: low' in Other_Information (after Evidence/Location). "
    #     "You may still merge mentions only when you are very confident they are the same underlying dataset; otherwise keep them separate. "
    #     "REMEMBER: Output ONLY a valid JSON array. Start with '[' and end with ']'. "
    #     "Do NOT include any explanations, comments, or natural language text. "
    #     "If no datasets are found, output an empty array: []."
    # )



    
    user_prompt = "\n\n".join(section_blocks)
    # 打印提示词会非常长；如需调试请设置环境变量 PRINT_PROMPT=1
    if os.getenv("PRINT_PROMPT") == "1":
        print(user_prompt)
    # 尝试调用 LLM
    all_think_content = []  # 收集所有重试中的思考内容
    parse_failed_info = {
        "first_attempt_failed": False,
        "all_attempts_failed": False,
        "error_message": None,
        "failed_attempts": 0,
    }
    first_attempt_error = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            content_raw = client.chat(user_prompt, system_prompt)
            content, think_content = _extract_think_content(content_raw)
            # print("="*30)
            # print(content_raw)
            if think_content:
                all_think_content.append(think_content)
                print("[INFO] 检测到 <think> 思考内容，已提取保存。")
            
            # 检查提取后的内容是否为空
            if not content or not content.strip():
                print(f"[WARN] 提取思考内容后，剩余内容为空。原始内容长度: {len(content_raw)}")
                print(f"[DEBUG] 原始内容前500字符: {content_raw[:500] if content_raw else 'None'}")
                # 如果内容为空，尝试从原始内容中提取 JSON（可能思考标签格式有问题）
                content = content_raw
                print(f"[INFO] 使用原始内容尝试解析 JSON...")
            
            # 尝试直接解析 JSON
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                # 如果直接解析失败，尝试从文本中提取 JSON
                print(f"[WARN] 直接解析 JSON 失败，尝试从文本中提取 JSON...")
                print(f"[DEBUG] JSON解析错误: {e}")
                print(f"[DEBUG] 内容总长度: {len(content)} 字符")
                print(f"[DEBUG] 前500字符: {content[:500] if content else 'None'}")
                print(f"[DEBUG] 后500字符: {content[-500:] if len(content) > 500 else content}")
                
                # 检查括号是否匹配（判断是否被截断）
                bracket_count = content.count('[') - content.count(']')
                brace_count = content.count('{') - content.count('}')
                if bracket_count != 0 or brace_count != 0:
                    print(f"[WARN] 括号不匹配！方括号差: {bracket_count}, 花括号差: {brace_count}")
                    print(f"[WARN] 可能原因: JSON被截断（max_tokens限制）或格式不完整")
                
                extracted_json = _extract_json_from_text(content)
                if extracted_json:
                    print(f"[INFO] 成功从文本中提取 JSON")
                    parsed = json.loads(extracted_json)
                else:
                    print(f"[ERROR] 无法从文本中提取有效的 JSON")
                    print(f"[DEBUG] LLM 返回内容前500字符: {content[:500] if content else 'None'}")
                    print(f"[DEBUG] LLM 返回内容后500字符: {content[-500:] if len(content) > 500 else content}")
                    
                    # 保存失败的响应到文件以便调试
                    # os 和 time 已在文件顶部导入，无需重复导入
                    debug_dir = Path(__file__).parent / "debug_failed_responses"
                    debug_dir.mkdir(exist_ok=True)
                    debug_file = debug_dir / f"failed_response_{int(time.time())}_{attempt}.txt"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== 完整响应内容 ===\n")
                        f.write(f"长度: {len(content)} 字符\n")
                        f.write(f"括号匹配: [ {content.count('[')} / ] {content.count(']')}, {{ {content.count('{')} / }} {content.count('}')}\n")
                        f.write(f"\n=== 内容 ===\n{content}\n")
                    print(f"[DEBUG] 完整响应已保存到: {debug_file}")
                    
                    error_msg = f"无法从文本中提取有效的 JSON"
                    if attempt == 1:
                        first_attempt_error = error_msg
                        parse_failed_info["first_attempt_failed"] = True
                        parse_failed_info["error_message"] = error_msg
                        # 保存原始输出以便调试
                        if content:
                            parse_failed_info["raw_output"] = content
                        elif 'content_raw' in locals() and content_raw:
                            parse_failed_info["raw_output"] = content_raw
                    parse_failed_info["failed_attempts"] += 1
                    raise json.JSONDecodeError(error_msg, content, 0)
            
            datasets = _normalize_llm_output(parsed, content)
            
            # 检查数据集是否为空
            if not datasets:
                print(f"[WARN] 解析后的数据集列表为空")
            
            # 合并所有思考内容（如果有多次重试，合并所有思考内容）
            final_think = "\n\n--- 重试分割线 ---\n\n".join(all_think_content) if all_think_content else None
            return datasets, final_think, parse_failed_info
        
        except (json.JSONDecodeError, KeyError, ValueError) as parse_exc:
            error_msg = str(parse_exc)
            if attempt == 1:
                first_attempt_error = error_msg
                parse_failed_info["first_attempt_failed"] = True
                parse_failed_info["error_message"] = error_msg
                # 保存原始输出以便调试（保存最后一次尝试的内容）
                if 'content' in locals() and content:
                    parse_failed_info["raw_output"] = content
                elif 'content_raw' in locals() and content_raw:
                    parse_failed_info["raw_output"] = content_raw
            parse_failed_info["failed_attempts"] += 1
            print(f"[ERROR] 解析 LLM 输出失败（第 {attempt} 次）：{parse_exc}")
            if 'content' in locals():
                print(f"[DEBUG] LLM 返回内容前500字符: {content[:500] if content else 'None'}")
            traceback.print_exc()
        except Exception as exc:
            error_msg = str(exc)
            if attempt == 1:
                first_attempt_error = error_msg
                parse_failed_info["first_attempt_failed"] = True
                parse_failed_info["error_message"] = error_msg
                # 保存原始输出以便调试（保存最后一次尝试的内容）
                if 'content' in locals() and content:
                    parse_failed_info["raw_output"] = content
                elif 'content_raw' in locals() and content_raw:
                    parse_failed_info["raw_output"] = content_raw
            parse_failed_info["failed_attempts"] += 1
            print(f"[ERROR] 调用 LLM 失败（第 {attempt} 次）：{exc}")
            traceback.print_exc()
        
        if attempt < max_attempts:
            sleep_time = base_delay * attempt
            print(f"[INFO] {sleep_time} 秒后重试...")
            time.sleep(sleep_time)
    
    # 如果所有重试都失败，返回空列表和收集到的思考内容
    parse_failed_info["all_attempts_failed"] = True
    if not parse_failed_info["error_message"] and first_attempt_error:
        parse_failed_info["error_message"] = first_attempt_error
    final_think = "\n\n--- 重试分割线 ---\n\n".join(all_think_content) if all_think_content else None
    return [], final_think, parse_failed_info


# ==================== 命令行工具 ====================

def main():
    """命令行测试工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SGLang LLM 客户端测试工具")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:30000",
        help="服务器地址（默认：http://127.0.0.1:30000）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="模型名称（默认：Qwen/Qwen3-4B-Instruct-2507）"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="测试连接"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="测试提示词"
    )
    
    args = parser.parse_args()
    
    # 创建客户端
    client = SGLangClient(
        base_url=args.base_url,
        model_name=args.model_name
    )
    
    # 测试连接
    if args.test_connection:
        if client.test_connection():
            print("[SUCCESS] 连接测试成功")
        else:
            print("[FAILED] 连接测试失败")
            sys.exit(1)
    
    # 测试对话
    if args.prompt:
        print(f"[INFO] 发送提示词: {args.prompt}")
        try:
            response = client.chat(args.prompt)
            print(f"[INFO] LLM 响应:\n{response}")
        except Exception as exc:
            print(f"[ERROR] 调用失败: {exc}")
            sys.exit(1)
    
    # 如果没有指定任何操作，默认测试连接和简单对话
    if not args.test_connection and not args.prompt:
        print("[INFO] 开始默认测试...")
        
        # 测试连接
        if not client.test_connection():
            print("[FAILED] 连接测试失败")
            sys.exit(1)
        
        # 测试对话
        test_prompt = "你好，请用一句话介绍你自己。"
        print(f"[INFO] 测试提示词: {test_prompt}")
        try:
            response = client.chat(test_prompt)
            print(f"[SUCCESS] LLM 响应:\n{response}")
        except Exception as exc:
            print(f"[ERROR] 调用失败: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()

