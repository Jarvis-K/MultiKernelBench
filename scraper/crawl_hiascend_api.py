#!/usr/bin/env python3
import os, re, json, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from urllib.parse import urljoin
import urllib.request

ROOT = "https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0003.html"
BASE = "https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/"

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def http_get(url: str, timeout: int = 20, retries: int = 3, backoff: float = 0.8) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** attempt))
    raise last_err


def extract_links(index_html: str):
    links = re.findall(r"href=\"(atlasascendc_api_\d+_\d+\.html)\"", index_html)
    # de-dup while preserving order
    seen = set()
    ordered = []
    for h in links:
        if h not in seen:
            seen.add(h)
            ordered.append(urljoin(BASE, h))
    return ordered


def extract_between(html: str, start_pat: str, end_pat: str) -> str:
    s = re.search(start_pat, html, flags=re.I|re.S)
    if not s:
        return ""
    start = s.end()
    sub = html[start:]
    e = re.search(end_pat, sub, flags=re.I|re.S)
    end = (start + e.start()) if e else len(html)
    return html[start:end]


def strip_tags(html: str) -> str:
    # remove code line numbers table cells and scripts/styles
    html = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", "", html, flags=re.I)
    html = re.sub(r"<td class=\"linenos\">[\s\S]*?</td>", "", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return unescape(html).strip()


def extract_section_text(html: str, title: str) -> str:
    # pattern for <h4 class="sectiontitle">功能说明</h4> ... next <h4 class="sectiontitle">
    pat_start = rf"<h4[^>]*class=\"sectiontitle\"[^>]*>\s*{re.escape(title)}\s*</h4>"
    # end at next h4 or end of document
    part = extract_between(html, pat_start, r"<h4[^>]*class=\"sectiontitle\"")
    return strip_tags(part)


def extract_h1(html: str) -> str:
    m = re.search(r"<h1[^>]*>\s*([^<]+)\s*</h1>", html, flags=re.I)
    if m:
        return unescape(m.group(1)).strip()
    # fallback to title prefix
    t = re.search(r"<title>([^<]+)</title>", html, flags=re.I)
    if t:
        return unescape(t.group(1).split("-")[0]).strip()
    return ""


def extract_code_block_after(html: str, title: str) -> str:
    # find the section, then grab first code block within
    pat_start = rf"<h4[^>]*class=\"sectiontitle\"[^>]*>\s*{re.escape(title)}\s*</h4>"
    s = re.search(pat_start, html, flags=re.I)
    if not s:
        return ""
    sub = html[s.end():]
    # drop line-number cells to avoid capturing only numbers
    sub = re.sub(r"<td class=\"linenos\">[\s\S]*?</td>", "", sub, flags=re.I)
    # code in <pre> or within tables labeled code
    # prefer code cell content if present
    m = re.search(r"<td[^>]*class=\"code\"[^>]*>[\s\S]*?<pre[^>]*>([\s\S]*?)</pre>", sub, flags=re.I)
    if not m:
        m = re.search(r"<pre[^>]*>([\s\S]*?)</pre>", sub, flags=re.I)
    if m:
        code = m.group(1)
        code = re.sub(r"<[^>]+>", "", code)
        return unescape(code).strip()
    # sometimes wrapped in codecoloring table
    m = re.search(r"<div class=\"codecoloring\"[\s\S]*?<pre[^>]*>([\s\S]*?)</pre>", sub, flags=re.I)
    if m:
        code = m.group(1)
        code = re.sub(r"<[^>]+>", "", code)
        return unescape(code).strip()
    return ""


def extract_params_table(html: str) -> str:
    # there may be two tables: 模板参数说明 and 参数说明; capture both
    out_parts = []
    for cap in ("模板参数说明", "参数说明"):
        # take nearest table after the <caption><b>表X </b>cap</caption>
        m = re.search(rf"<caption>\s*<b>表\d+\s*</b>\s*{cap}\s*</caption>[\s\S]*?</table>", html, flags=re.I)
        if m:
            out_parts.append(strip_tags(m.group(0)))
    return "\n\n".join(out_parts)


def parse_api_page(url: str, html: str):
    record = {
        "API名称": None,
        "API文档URL": url,
        "功能说明": None,
        "函数原型": None,
        "参数说明": None,
        "返回值": None,
        "调用示例": None,
        "错误": None,
    }
    try:
        record["API名称"] = extract_h1(html)
        record["功能说明"] = extract_section_text(html, "功能说明")
        record["函数原型"] = extract_code_block_after(html, "函数原型") or extract_section_text(html, "函数原型")
        # 参数说明来自两类表格
        record["参数说明"] = extract_params_table(html) or extract_section_text(html, "参数说明")
        record["返回值"] = extract_section_text(html, "返回值")
        record["调用示例"] = extract_code_block_after(html, "调用示例") or extract_section_text(html, "调用示例")
    except Exception as e:
        record["错误"] = f"解析异常: {e}"
    return record


def main():
    os.makedirs("/workspace/scraper/out", exist_ok=True)
    # load index
    index_html = open("/workspace/scraper/root.html", "r", encoding="utf-8", errors="ignore").read()
    urls = extract_links(index_html)
    results = [None] * len(urls)
    def worker(idx_url):
        idx, url = idx_url
        try:
            html = http_get(url)
            rec = parse_api_page(url, html)
            missing = [k for k in ("API名称","功能说明","函数原型","参数说明","返回值") if not rec.get(k)]
            if missing:
                joined = ",".join(missing)
                prev = rec.get("错误") or ""
                rec["错误"] = (prev + (" " if prev else "") + f"缺失字段: {joined}").strip()
        except Exception as e:
            rec = {
                "API名称": None,
                "API文档URL": url,
                "功能说明": None,
                "函数原型": None,
                "参数说明": None,
                "返回值": None,
                "调用示例": None,
                "错误": f"抓取失败: {e}",
            }
        results[idx] = rec

    with ThreadPoolExecutor(max_workers=min(16, max(4, os.cpu_count() or 4))) as ex:
        for _ in ex.map(worker, list(enumerate(urls))):
            pass
    # write json
    out_path = "/workspace/scraper/out/ascendc_api.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(out_path)

if __name__ == "__main__":
    main()
