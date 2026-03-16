"""
使用本地 SGLang/OpenAI 兼容 API 复用 data_meta_extract.py 的数据元信息提取流程。

示例：
    python data_local/LLM.py --articles-dir 'Articles/Nature Cities/2025-article' --base-url http://127.0.0.1:30000 --model-name Qwen/Qwen3-4B-Instruct-2507

也可以单独测试对话：
    python data_local/LLM.py --prompt "你好"

新增能力：
- --save-paper：将输入给 LLM 的拼接全文落盘为 paper.txt（默认写到每篇 article.json 所在目录）
- --include-full-references：拼接全文时包含完整 References（content+links），更接近 auto_benchmark 的 stage1_generate.py 行为
"""

import argparse
import json
import sys
import time
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from sglang_LLM import SGLangClient, extract_metadata_with_sglang


def format_time(seconds):
    """将秒数格式化为可读的时间字符串"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}小时{minutes}分{secs}秒"


# 导入检查功能（可选）
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from meta_data_check import check_meta_data
    CHECK_AVAILABLE = True
except ImportError:
    CHECK_AVAILABLE = False


def extract_data_availability(json_data: Dict) -> Optional[Dict[str, Union[str, List[str]]]]:
    """
    扩展版 Data Availability 解析逻辑，支持从多个位置查找，并支持多种类似名称。
    
    按以下优先级查找：
    1. Extra_info 中的相关键（兼容旧逻辑）
    2. Sections 数组中 title/heading/name 含关键字的 section
    3. 顶层字段名中含关键字的字段
    
    支持的关键词：data availability, availability of data, data access, data-access,
    data and code availability, availability of data and materials
    """
    KEYWORDS = [
        "data availability",
        "availability of data",
        "data access",
        "data-access",
        "data and code availability",
        "availability of data and materials",
    ]
    
    def _match_keyword(name: str) -> bool:
        """判断字段名是否包含 data availability 相关关键字（大小写不敏感，子串匹配）。"""
        if not isinstance(name, str):
            return False
        lower = name.lower()
        return any(kw in lower for kw in KEYWORDS)
    
    def _extract_links(value: Any) -> List[str]:
        """从 value 中尽量提取 links，兼容 dict/list/dict-of-links。"""
        links: List[str] = []
        if not isinstance(value, dict):
            return links
        raw = value.get("links") or value.get("Links") or []
        if isinstance(raw, dict):
            # 变成 "key: url" 形式的字符串列表，方便后续使用
            links = [f"{k}: {v}" for k, v in raw.items()]
        elif isinstance(raw, list):
            links = [str(x) for x in raw]
        return links
    
    # 1) 优先在 Extra_info 中找（兼容原来逻辑）
    extra_info = json_data.get("Extra_info")
    if isinstance(extra_info, dict):
        for key, value in extra_info.items():
            if not _match_keyword(key):
                continue
            # 内容优先从 content / text / value 取，退化为字符串本身
            content = None
            if isinstance(value, dict):
                content = (
                    value.get("content")
                    or value.get("text")
                    or value.get("value")
                )
                if isinstance(content, str):
                    content = content.strip() or None
            elif isinstance(value, str):
                content = value.strip() or None

            links = _extract_links(value)
            if content or links:
                return {"content": content, "links": links}

    # 2) 在 Sections 里找 title/heading/name 含关键字的小节
    sections = json_data.get("Sections")
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            title = None
            for tk in ("title", "heading", "name"):
                tv = sec.get(tk)
                if isinstance(tv, str) and tv.strip():
                    title = tv.strip()
                    break
            if not title or not _match_keyword(title):
                continue

            # 正文优先取 text / content / value
            text_value = sec.get("text") or sec.get("content") or sec.get("value")
            content = None
            if isinstance(text_value, str):
                content = text_value.strip() or None
            elif isinstance(text_value, dict):
                content = (
                    text_value.get("content")
                    or text_value.get("text")
                    or text_value.get("value")
                )
                if isinstance(content, str):
                    content = content.strip() or None

            if content:
                # Sections 里一般没有 links，这里统一返回空列表
                return {"content": content, "links": []}

    # 3) 顶层字段名中含关键字的情况（排除 Extra_info / Sections 自身）
    for key, value in json_data.items():
        if key in ("Extra_info", "Sections"):
            continue
        if not _match_keyword(key):
            continue

        content = None
        if isinstance(value, str):
            content = value.strip() or None
        elif isinstance(value, dict):
            content = (
                value.get("content")
                or value.get("text")
                or value.get("value")
            )
            if isinstance(content, str):
                content = content.strip() or None

        links: List[str] = []
        if isinstance(value, dict):
            links = _extract_links(value)

        if content or links:
            return {"content": content, "links": links}

    # 如果都没找到，返回 None
    return None


def _stringify_extra_info(extra_info: Any) -> str:
    if not isinstance(extra_info, dict):
        return ""
    parts: List[str] = []
    # 输出 Extra_info 的内容与 links（不要丢 URL；保持 key 的原始顺序）
    for k, v in extra_info.items():
        if not isinstance(k, str):
            continue
        content = None
        links: Any = None
        if isinstance(v, dict):
            content = v.get("content") or v.get("text") or v.get("value")
            links = v.get("links") or v.get("Links")
        elif isinstance(v, str):
            content = v

        block_parts: List[str] = []
        if isinstance(content, str) and content.strip():
            block_parts.append(content.strip())

        # links 可能是 dict 或 list；统一转成文本
        if isinstance(links, dict) and links:
            link_lines = [f"- {lk}: {lv}" for lk, lv in links.items()]
            block_parts.append("Links:\n" + "\n".join(link_lines))
        elif isinstance(links, list) and links:
            block_parts.append("Links:\n" + "\n".join(f"- {item}" for item in links))

        if block_parts:
            parts.append(f"## {k}\n" + "\n".join(block_parts))

    return "\n\n".join(parts)


def _looks_corrupted(text: str) -> bool:
    """粗略判断 text 是否存在明显拼接/截断痕迹。"""
    t = text or ""
    if not t:
        return False
    if re.search(r"\.\s+[a-z]{2,}", t):
        if re.search(r"\.\s+luded\b", t):
            return True
    if re.search(r"\b[a-z]+[A-Z][a-z]+\b", t):
        return True
    if "incocioeconomic" in t or "sIndicators" in t:
        return True
    return False


def _section_text_from_sentences(sec: Dict[str, Any]) -> Optional[str]:
    """当 section.text 质量较差时，用 sentences 重建正文。"""
    paragraphs = sec.get("paragraphs")
    if not isinstance(paragraphs, list):
        return None
    out_paras: List[str] = []
    for p in paragraphs:
        if not isinstance(p, dict):
            continue
        sentences = p.get("sentences")
        if isinstance(sentences, list) and sentences:
            ss: List[str] = []
            for s in sentences:
                if not isinstance(s, dict):
                    continue
                st = s.get("text")
                if isinstance(st, str) and st.strip():
                    ss.append(st.strip())
            if ss:
                out_paras.append(" ".join(ss))
                continue
        pt = p.get("text")
        if isinstance(pt, str) and pt.strip():
            out_paras.append(pt.strip())
    if not out_paras:
        return None
    return "\n".join(out_paras)


def _is_noisy_section(title: str) -> bool:
    """判定是否为噪音章节（致谢、作者信息、版权声明等）"""
    if not title:
        return False
    title_lower = title.lower()
    noise_keywords = [
        "acknowledgement", "author information", "author contribution", "corresponding author",
        "ethics declaration", "competing interest", "consent", "publisher’s note",
        "supplementary information", "rights and permissions", "reprints and permissions",
        "about this article", "peer review information", "reporting summary",
        "data protection statement", "competing financial interest"
    ]
    return any(kw in title_lower for kw in noise_keywords)


def build_full_article_prompt(article: Dict, include_full_references: bool = False) -> str:
    """
    将 article.json 的结构拼接为可读全文（同步 stage1_generate.py 逻辑）：
    - Title
    - Abstract
    - Sections: 每个 section 用二级标题包裹（自动过滤噪音章节）
    - Extra_info: Data availability / Code availability（如存在）
    - Figures / References
    """
    chunks: List[str] = []

    title = article.get("title") or article.get("Title")
    if isinstance(title, str) and title.strip():
        chunks.append(f"# Title\n{title.strip()}")

    abstract = article.get("Abstract") or article.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        chunks.append(f"# Abstract\n{abstract.strip()}")

    sections = article.get("Sections")
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_title = (sec.get("title") or sec.get("heading") or sec.get("name") or "").strip()
            
            # 过滤噪音章节
            if _is_noisy_section(sec_title):
                continue

            sec_text = sec.get("text") or sec.get("content")

            # 优先用更可靠的结构重建文本
            if isinstance(sec_text, str) and sec_text.strip():
                if _looks_corrupted(sec_text):
                    rebuilt = _section_text_from_sentences(sec)
                    if isinstance(rebuilt, str) and rebuilt.strip():
                        sec_text = rebuilt

            if isinstance(sec_title, str) and sec_title.strip() and isinstance(sec_text, str) and sec_text.strip():
                block = [f"# {sec_title.strip()}", sec_text.strip()]

                # 保留该 section 的 cites/figures/links
                cites = sec.get("cites")
                if isinstance(cites, list) and cites:
                    cites_str = ", ".join(str(x) for x in cites if isinstance(x, str) and x.strip())
                    if cites_str:
                        block.append(f"Section_Cites: {cites_str}")

                links = sec.get("links") or sec.get("Links")
                if isinstance(links, dict) and links:
                    link_lines = [f"- {lk}: {lv}" for lk, lv in links.items()]
                    block.append("Section_Links:\n" + "\n".join(link_lines))

                figs = sec.get("figures")
                if isinstance(figs, list) and figs:
                    figs_str = ", ".join(str(x) for x in figs if isinstance(x, str) and x.strip())
                    if figs_str:
                        block.append(f"Section_Figures: {figs_str}")

                chunks.append("\n".join(block))

    extra_txt = _stringify_extra_info(article.get("Extra_info"))
    if extra_txt:
        chunks.append("# Extra_info\n" + extra_txt)

    # 追加 Figures
    figures = article.get("Figures")
    if isinstance(figures, list) and figures:
        fig_lines: List[str] = []
        for f in figures:
            if not isinstance(f, dict): continue
            fid = f.get("id")
            ftitle = f.get("figure-title") or f.get("title")
            flink = f.get("figure-link") or f.get("link")
            if isinstance(fid, str) and fid.strip():
                line = f"- {fid.strip()}"
                if isinstance(ftitle, str) and ftitle.strip(): line += f": {ftitle.strip()}"
                if isinstance(flink, str) and flink.strip(): line += f" ({flink.strip()})"
                fig_lines.append(line)
        if fig_lines:
            chunks.append("# Figures\n" + "\n".join(fig_lines))

    # 追加 References (同步参数控制逻辑)
    references = article.get("References")
    if isinstance(references, list) and references:
        ref_lines: List[str] = []
        for r in references:
            if not isinstance(r, dict): continue
            rid = r.get("id")
            rcontent = r.get("content")
            if not (isinstance(rid, str) and rid.strip()): continue
            
            if include_full_references:
                line = f"- {rid.strip()}: {rcontent.strip() if isinstance(rcontent, str) else ''}"
                ref_lines.append(line)
                rlinks = r.get("links") or r.get("Links")
                if isinstance(rlinks, dict) and rlinks:
                    for lk, lv in rlinks.items():
                        ref_lines.append(f"  - {lk}: {lv}")
            else:
                # 即使不包含全文，也保留 ID 供参考
                ref_lines.append(f"[{rid.strip()}]")
                
        if ref_lines:
            title_str = "# References (full)" if include_full_references else "# References (IDs only)"
            chunks.append(f"{title_str}\n" + "\n".join(ref_lines))

    if not chunks:
        chunks.append(json.dumps(article, ensure_ascii=False, indent=2))
    return "\n\n".join(chunks).strip() + "\n"


def _extract_text_from_section(value: Union[None, str, Dict]) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, dict):
        content = value.get("content", "")
        links = value.get("links") or value.get("Links") or {}
        text_parts = [content.strip()] if content else []
        if isinstance(links, dict) and links:
            link_lines = [f"- {k}: {v}" for k, v in links.items()]
            text_parts.append("Links:\n" + "\n".join(link_lines))
        elif isinstance(links, list) and links:
            text_parts.append("Links:\n" + "\n".join(str(item) for item in links))
        merged = "\n".join(part for part in text_parts if part)
        return merged or None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def extract_methods_section(json_data: Dict) -> Optional[str]:
    """提取 Methods 部分内容，包括文本和参考文献。
    
    按以下优先级查找：
    1. 顶层字段（如 "Methods"）
    2. Extra_info 中的 "Methods" 键
    3. Sections 数组中 title 为 "Methods" 的 section 的 text 字段
    
    如果找到 Methods section，还会提取其 cites 字段对应的完整参考文献内容。
    """
    def _candidate_text(entry: Optional[Union[str, Dict]]) -> Optional[str]:
        return _extract_text_from_section(entry)
    
    def _get_references_by_ids(json_data: Dict, cite_ids: List[str]) -> List[str]:
        """根据引用ID列表从 References 数组中提取完整的参考文献内容"""
        references = json_data.get("References", [])
        if not isinstance(references, list):
            return []
        
        # 创建引用ID到内容的映射
        ref_map = {}
        for ref in references:
            if isinstance(ref, dict):
                ref_id = ref.get("id")
                ref_content = ref.get("content", "")
                if ref_id and ref_content:
                    ref_map[ref_id] = ref_content
        
        # 按 cite_ids 的顺序提取参考文献
        ref_contents = []
        for cite_id in cite_ids:
            if cite_id in ref_map:
                ref_contents.append(f"[{cite_id}] {ref_map[cite_id]}")
        
        return ref_contents

    methods_text = None
    methods_cites = None

    # 优先检查 Sections 数组（因为可以获取 cites 信息）
    sections = json_data.get("Sections", [])
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                title = section.get("title", "")
                if isinstance(title, str) and "method" in title.lower():
                    text = section.get("text", "")
                    if isinstance(text, str) and text.strip():
                        methods_text = text.strip()
                        # 提取 cites 字段
                        cites = section.get("cites", [])
                        if isinstance(cites, list) and cites:
                            methods_cites = cites
                        break
    
    # 如果 Sections 中没有找到，再检查其他位置
    if not methods_text:
        # 1. 检查顶层字段
        for key, value in json_data.items():
            if isinstance(key, str) and "method" in key.lower():
                text = _candidate_text(value)
                if text:
                    methods_text = text
                    break

        # 2. 检查 Extra_info
        if not methods_text:
            extra_info = json_data.get("Extra_info", {})
            if isinstance(extra_info, dict):
                for key, value in extra_info.items():
                    if isinstance(key, str) and "method" in key.lower():
                        text = _candidate_text(value)
                        if text:
                            methods_text = text
                            break
    
    # 如果没有找到 methods_text，返回 None
    if not methods_text:
        return None
    
    # 如果找到了 methods_text 和 cites，提取对应的参考文献
    result_parts = [methods_text]
    
    if methods_cites:
        ref_contents = _get_references_by_ids(json_data, methods_cites)
        if ref_contents:
            result_parts.append("\n\nReferences cited in Methods section:")
            result_parts.extend(ref_contents)
    
    return "\n".join(result_parts)


def build_meta_output(article: Dict, datasets: List[Dict], think_content: Optional[str] = None, parse_failed_info: Optional[Dict] = None) -> Dict:
    output = {
        "id": article.get("id"),
        "title": article.get("title"),
        "journal": article.get("journal"),
        "pdf_link": article.get("pdf_link"),
        "open_access": article.get("open_access"),
        "meta_data": datasets,
    }
    if think_content:
        output["think"] = think_content
    
    # 如果第一次解析失败，添加标记字段
    if parse_failed_info and parse_failed_info.get("first_attempt_failed", False):
        output["parse_failed"] = True
        if parse_failed_info.get("error_message"):
            output["parse_error"] = parse_failed_info["error_message"]
        from datetime import datetime
        output["parse_failed_at"] = datetime.now().isoformat()
        if parse_failed_info.get("failed_attempts", 0) > 0:
            output["parse_failed_attempts"] = parse_failed_info["failed_attempts"]
        # 保存原始输出以便调试
        if parse_failed_info.get("raw_output"):
            output["raw_output"] = parse_failed_info["raw_output"]
    
    return output


def process_article(
    json_path: Path,
    client: SGLangClient,
    overwrite: bool = False,
    verbose: bool = True,
    max_attempts: int = 3,
    base_delay: int = 5,
    enable_check: bool = False,
    check_max_retries: int = 2,
    skip_processed: bool = False,
    test_mode: bool = False,
    model_name: str = "",
    save_paper: bool = False,
    paper_filename: str = "paper.txt",
    include_full_references: bool = False,
) -> Tuple[Dict[str, int], Optional[Dict]]:
    stats = {"processed": 0, "skipped": 0, "error": 0, "regenerated": 0}
    parse_failed_info = None

    # 方案A：如果启用跳过已处理文章，且 meta_data.json 存在且未设置 overwrite，直接跳过
    if skip_processed:
        meta_path = json_path.parent / "meta_data.json"
        if meta_path.exists() and not overwrite:
            if verbose:
                print(f"[SKIP] {json_path}: meta_data.json 已存在，跳过（使用 --overwrite 强制重新处理）")
            stats["skipped"] = 1
            return stats, None

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        if verbose:
            print(f"[ERROR] 无法读取 {json_path}: {exc}")
        stats["error"] = 1
        return stats, None

    # Phase-1：获取全文（优先从 paper.txt 读取，否则从 article.json 拼接）
    paper_path = json_path.parent / (paper_filename or "paper.txt")
    full_article_text = None
    
    if paper_path.exists():
        try:
            full_article_text = paper_path.read_text(encoding="utf-8")
            if verbose:
                print(f"[OK] 直接从 paper.txt 导入全文: {paper_path}")
        except Exception as exc:
            if verbose:
                print(f"[WARN] 读取 paper.txt 失败，将重新从 article.json 拼接: {exc}")

    if not full_article_text:
        # - include_full_references=False：旧行为（仅保留 References IDs，减少 token）
        # - include_full_references=True ：更接近 stage1_generate.py，会包含 references 的 content + links
        full_article_text = build_full_article_prompt(data, include_full_references=include_full_references)
        
        # 可选：把输入给 LLM 的全文落盘，便于人工审计/排查（对齐 stage1_generate.py 的 paper.txt）
        if save_paper and full_article_text:
            try:
                if (not paper_path.exists()) or overwrite:
                    paper_path.write_text(full_article_text, encoding="utf-8")
                    if verbose:
                        print(f"[OK] 已写出 paper: {paper_path}")
                else:
                    if verbose:
                        print(f"[SKIP] paper 已存在，跳过（使用 --overwrite 强制覆盖）: {paper_path}")
            except Exception as exc:
                # paper.txt 写入失败不应影响主流程（meta_data.json 仍然要生成）
                if verbose:
                    print(f"[WARN] 写入 paper 失败（不影响主流程）：{exc}")

    if not full_article_text:
        if verbose:
            print(f"[WARN] {json_path}: 无法构建全文提示，跳过")
        stats["skipped"] = 1
        return stats, None

    if verbose:
        print(f"[INFO] 正在处理：{json_path}")
    
    # 生成数据集元数据（支持检查模式下的重试）
    datasets = None
    think_content = None
    parse_failed_info = None
    all_think_contents = []  # 收集所有重试中的思考内容
    check_retries = 0
    while True:
        try:
            datasets, current_think, parse_failed_info = extract_metadata_with_sglang(
                "",
                None,
                None,
                client=client,
                max_attempts=max_attempts,
                base_delay=base_delay,
                full_article_text=full_article_text,
            )
            # 收集思考内容
            if current_think:
                all_think_contents.append(current_think)
        except Exception as exc:
            if verbose:
                print(f"[ERROR] 调用本地 LLM 失败：{exc}")
            stats["error"] = 1
            return stats, parse_failed_info
        
        # 如果启用了检查，验证生成的数据
        if enable_check and CHECK_AVAILABLE:
            errors = check_meta_data(datasets, data.get("id"))
            if errors:
                if verbose:
                    print(f"[WARN] 发现 {len(errors)} 个低级错误：")
                    for err in errors[:3]:  # 只显示前3个错误
                        print(f"  - {err}")
                    if len(errors) > 3:
                        print(f"  ... 还有 {len(errors) - 3} 个错误")
                
                # 如果还有重试次数，重新生成
                if check_retries < check_max_retries:
                    check_retries += 1
                    if verbose:
                        print(f"[INFO] 尝试重新生成（第 {check_retries}/{check_max_retries} 次）...")
                    continue
                else:
                    if verbose:
                        print(f"[WARN] 已达到最大重试次数，保留当前结果")
            else:
                # 没有错误，退出循环
                if check_retries > 0 and verbose:
                    print(f"[OK] 重新生成后无错误")
                break
        else:
            # 未启用检查，直接退出循环
            break

    # 合并所有思考内容（如果有多次重试，合并所有思考内容）
    if all_think_contents:
        think_content = "\n\n--- 检查重试分割线 ---\n\n".join(all_think_contents)

    # 确保 datasets 不为 None
    if datasets is None:
        if verbose:
            print(f"[WARN] datasets 为 None，使用空列表")
        datasets = []
    
    if verbose:
        print(f"[DEBUG] 准备保存：datasets 数量={len(datasets) if datasets else 0}, think_content={'有' if think_content else '无'}")
        if parse_failed_info and parse_failed_info.get("first_attempt_failed", False):
            print(f"[WARN] 第一次解析失败，将在 meta_data.json 中添加标记")

    meta_output = build_meta_output(data, datasets, think_content, parse_failed_info)
    meta_path = json_path.parent / "meta_data.json"

    if meta_path.exists() and not overwrite:
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        existing.update(meta_output)
        meta_output = existing
    
    # 添加模型元信息
    meta_output["_meta"] = {
        "model_name": client.model_name,
        "processed_at": datetime.now().isoformat()
    }

    try:
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_output, f, ensure_ascii=False, indent=4)
        if verbose:
            print(f"[OK] 已生成元数据：{meta_path}")
        stats["processed"] = 1
        if check_retries > 0:
            stats["regenerated"] = 1
        
        # ========== 测试模式：额外保存到 golden_set/test 目录（独立代码块，后续可删除）==========
        if test_mode and model_name:
            try:
                # 获取文章ID
                article_id = data.get("id", "")
                if not article_id:
                    # 如果文章ID不存在，使用文件夹名
                    article_id = json_path.parent.name
                
                # 构建测试目录路径：Benchmark/golden_set/test/{模型名}/{文章ID}/
                # 模型名可能包含斜杠，需要替换为安全字符
                safe_model_name = model_name.replace("/", "_").replace("\\", "_")
                test_base_dir = Path("/data1/yourunwen/Workspace/data_local_check/Benchmark/golden_set/test")
                test_model_dir = test_base_dir / safe_model_name
                test_article_dir = test_model_dir / article_id
                test_article_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存 meta_data.json
                test_meta_path = test_article_dir / "meta_data.json"
                with test_meta_path.open("w", encoding="utf-8") as f:
                    json.dump(meta_output, f, ensure_ascii=False, indent=4)
                
                if verbose:
                    print(f"[TEST] 已额外保存到测试目录：{test_meta_path}")
            except Exception as test_exc:
                if verbose:
                    print(f"[WARN] 测试模式保存失败：{test_exc}")
        # ========== 测试模式代码块结束 ==========
        
    except Exception as exc:
        if verbose:
            print(f"[ERROR] 写入 {meta_path} 失败：{exc}")
        stats["error"] = 1
    
    # 如果解析失败，添加失败时间戳
    if parse_failed_info and parse_failed_info.get("first_attempt_failed", False):
        parse_failed_info["failed_at"] = datetime.now().isoformat()
    
    return stats, parse_failed_info


def collect_article_paths(articles_dir: Optional[str], single_article: Optional[str], limit: Optional[int]) -> List[Path]:
    paths: List[Path] = []
    if single_article:
        candidate = Path(single_article)
        if not candidate.exists():
            raise FileNotFoundError(f"指定的 article.json 不存在：{candidate}")
        paths.append(candidate)
        return paths

    if not articles_dir:
        return paths

    base_dir = Path(articles_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"目录不存在：{base_dir}")

    paths = sorted(base_dir.rglob("article.json"))
    if limit:
        paths = paths[:limit]
    return paths


def main():
    parser = argparse.ArgumentParser(description="使用本地 SGLang API 批量提取文章数据集元信息")
    parser.add_argument("--articles-dir", type=str, help="待处理文章的根目录")
    parser.add_argument("--article-json", type=str, help="单个 article.json 文件路径")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000", help="本地 API 地址（无需带 /v1）")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="模型名称")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key（本地部署通常无需修改）")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--max-tokens", type=int, default=60000, help="最大生成 token 数")
    parser.add_argument("--timeout", type=int, default=600, help="请求超时时间（秒）")
    parser.add_argument("--overwrite", action="store_true", help="若 meta_data.json 已存在则覆盖")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的文章数量")
    parser.add_argument("--max-attempts", type=int, default=3, help="LLM 调用最大重试次数")
    parser.add_argument("--retry-delay", type=int, default=5, help="失败后的基础重试间隔（秒）")
    parser.add_argument("--enable-check", action="store_true", help="启用边生成边检查模式（发现错误立即重新生成）")
    parser.add_argument("--check-max-retries", type=int, default=2, help="检查模式下最大重试次数（默认：2）")
    parser.add_argument("--quiet", action="store_true", help="安静模式，减少日志输出")
    parser.add_argument("--prompt", type=str, help="测试对话提示词")
    parser.add_argument("--output-failed-log", type=str, default="parse_failed_articles.json", help="输出解析失败文章汇总日志的文件路径")
    parser.add_argument("--skip-processed", action="store_true", help="自动跳过已有 meta_data.json 的文章")
    parser.add_argument("--checkpoint-file", type=str, default=".resume_checkpoint.json", help="检查点文件路径")
    parser.add_argument("--no-resume", action="store_true", help="禁用检查点自动恢复")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--threads", type=int, default=1, help="并发处理线程数")
    parser.add_argument("--save-paper", action="store_true", help="将输入给 LLM 的拼接全文落盘为 paper.txt（写入每篇文章目录）")
    parser.add_argument("--paper-filename", type=str, default="paper.txt", help="落盘 paper 文件名（默认：paper.txt）")
    parser.add_argument("--include-full-references", action="store_true", help="拼接全文时包含完整 References（content+links），更接近 stage1_generate.py 的行为")
    args = parser.parse_args()

    if not any([args.prompt, args.articles_dir, args.article_json]):
        parser.error("必须通过 --articles-dir/--article-json 指定输入，或通过 --prompt 测试对话。")

    client = SGLangClient(
        base_url=args.base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    print("=" * 60)
    print("[模型配置]")
    print(f"  API地址: {args.base_url}")
    print(f"  模型名称: {args.model_name}")
    print(f"  温度: {args.temperature}")
    print(f"  最大Token数: {args.max_tokens}")
    print(f"  并发线程数: {args.threads}")
    print("=" * 60)

    if args.prompt:
        print("[INFO] 测试对话模式：")
        try:
            reply = client.chat(args.prompt)
            print(f"[LLM] {reply}")
        except Exception as exc:
            print(f"[ERROR] 对话失败：{exc}")
            if not (args.articles_dir or args.article_json):
                sys.exit(1)

    article_paths = collect_article_paths(args.articles_dir, args.article_json, args.limit)
    if not article_paths:
        if args.articles_dir or args.article_json:
            print("[WARN] 未找到任何 article.json")
        return

    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else None
    resume_index = 0
    original_total = len(article_paths)
    if checkpoint_path and checkpoint_path.exists() and not args.no_resume:
        try:
            with checkpoint_path.open("r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            last_path = checkpoint.get("last_processed_path")
            if last_path:
                for idx, path in enumerate(article_paths):
                    if str(path) == last_path:
                        resume_index = idx + 1
                        break
                if resume_index > 0:
                    print(f"[INFO] 从检查点恢复，从第 {resume_index + 1}/{original_total} 篇文章开始")
                    article_paths = article_paths[resume_index:]
        except Exception as exc:
            print(f"[WARN] 读取检查点文件失败: {exc}")

    total_stats = {"processed": 0, "skipped": 0, "error": 0, "regenerated": 0}
    failed_articles = []
    total = len(article_paths)
    start_time = time.time()
    
    # 线程安全锁
    lock = threading.Lock()

    def worker(idx, path):
        res_stats, res_failed_info = process_article(
            path,
            client=client,
            overwrite=args.overwrite,
            verbose=not args.quiet,
            max_attempts=args.max_attempts,
            base_delay=args.retry_delay,
            enable_check=args.enable_check,
            check_max_retries=args.check_max_retries,
            skip_processed=args.skip_processed,
            test_mode=args.test,
            model_name=args.model_name,
            save_paper=args.save_paper,
            paper_filename=args.paper_filename,
            include_full_references=args.include_full_references,
        )
        
        with lock:
            for key in total_stats:
                total_stats[key] += res_stats[key]
            
            processed_so_far = total_stats["processed"] + total_stats["skipped"] + total_stats["error"]
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / processed_so_far if processed_so_far > 0 else 0
            remaining = total - processed_so_far
            est_remaining = avg_time * remaining
            
            if not args.quiet:
                progress = (processed_so_far * 100) // total
                print(f"[进度] {processed_so_far}/{total} ({progress}%) | 已用: {format_time(elapsed_time)} | 剩余预计: {format_time(est_remaining)}")

            if res_failed_info and res_failed_info.get("first_attempt_failed", False):
                failed_articles.append({
                    "path": str(path),
                    "error": res_failed_info.get("error_message", "未知错误")
                })
            
            if checkpoint_path and res_stats.get("processed", 0) > 0:
                try:
                    checkpoint_data = {
                        "last_processed_path": str(path),
                        "created_at": datetime.now().isoformat(),
                    }
                    with checkpoint_path.open("w", encoding="utf-8") as f:
                        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                except:
                    pass

    if args.threads > 1:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(worker, i, p) for i, p in enumerate(article_paths, 1)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] 线程异常: {e}")
    else:
        for i, p in enumerate(article_paths, 1):
            worker(i, p)

    total_elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[INFO] 处理完成。总计: {total}, 已处理: {total_stats['processed']}, 跳过: {total_stats['skipped']}, 错误: {total_stats['error']}")
    print(f"[时间] 总耗时: {format_time(total_elapsed_time)}")
    print("=" * 60)

    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

if __name__ == "__main__":
    main()
