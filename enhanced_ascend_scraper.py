#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华为昇腾API文档增强版爬虫
深度爬取每个API页面的详细信息，提供更完整的数据提取
"""

import requests
import json
import re
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Set
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAscendAPIScraper:
    def __init__(self, base_url: str, max_workers: int = 5):
        self.base_url = base_url
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.visited_urls: Set[str] = set()
        self.failed_urls: Dict[str, str] = {}
        self.api_data: List[Dict] = []
        self.lock = threading.Lock()
        
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """获取页面内容"""
        try:
            logger.info(f"正在访问: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            error_msg = f"访问页面失败: {str(e)}"
            logger.error(f"{url} - {error_msg}")
            with self.lock:
                self.failed_urls[url] = error_msg
            return None
    
    def extract_api_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """提取API相关链接"""
        links = set()
        
        # 查找所有链接
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
                
            # 构建完整URL
            full_url = urljoin(base_url, href)
            
            # 过滤API相关链接
            if self.is_api_link(href, link.get_text()):
                links.add(full_url)
        
        # 查找导航菜单中的链接
        nav_selectors = [
            'nav', '[class*="nav"]', '[class*="menu"]', '[class*="sidebar"]',
            '[class*="toc"]', '[id*="nav"]', '[id*="menu"]'
        ]
        
        for selector in nav_selectors:
            nav_elements = soup.select(selector)
            for nav in nav_elements:
                for link in nav.find_all('a', href=True):
                    href = link.get('href')
                    if href and self.is_api_link(href, link.get_text()):
                        full_url = urljoin(base_url, href)
                        links.add(full_url)
        
        return list(links)
    
    def is_api_link(self, href: str, text: str) -> bool:
        """判断是否为API相关链接"""
        # 过滤无关链接
        exclude_patterns = [
            r'javascript:', r'mailto:', r'#', r'\.pdf$', r'\.doc$', r'\.zip$',
            r'download', r'login', r'register', r'search'
        ]
        
        href_lower = href.lower()
        for pattern in exclude_patterns:
            if re.search(pattern, href_lower):
                return False
        
        # API相关的URL模式
        api_patterns = [
            r'api.*\.html',
            r'ascendcopapi',
            r'atlasascendc_api',
            r'function',
            r'interface',
            r'method',
            r'operator'
        ]
        
        # API相关的文本模式
        text_patterns = [
            r'API',
            r'接口',
            r'函数',
            r'方法',
            r'算子',
            r'操作符',
            r'指令'
        ]
        
        text_lower = text.lower() if text else ''
        
        # 检查URL模式
        for pattern in api_patterns:
            if re.search(pattern, href_lower):
                return True
                
        # 检查文本模式
        for pattern in text_patterns:
            if re.search(pattern, text_lower, re.I):
                return True
                
        return False
    
    def extract_detailed_api_info(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """深度提取单个API页面的详细信息"""
        try:
            api_info = {
                "API名称": "",
                "API文档URL": url,
                "功能说明": "",
                "函数原型": "",
                "参数说明": [],
                "返回值": "",
                "调用示例": "",
                "约束限制": "",
                "相关接口": [],
                "版本信息": ""
            }
            
            # 提取页面标题
            title = soup.find('title')
            if title:
                api_info["API名称"] = self.clean_text(title.get_text())
            
            # 查找主要内容区域
            content_selectors = [
                'main', '[class*="content"]', '[class*="main"]', 
                '[id*="content"]', '[id*="main"]', 'article', '.document'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body')
            
            if main_content:
                # 提取API名称（如果标题中没有）
                if not api_info["API名称"]:
                    api_info["API名称"] = self.extract_api_name_from_content(main_content)
                
                # 提取功能说明
                api_info["功能说明"] = self.extract_detailed_description(main_content)
                
                # 提取函数原型
                api_info["函数原型"] = self.extract_detailed_function_prototype(main_content)
                
                # 提取参数说明
                api_info["参数说明"] = self.extract_detailed_parameters(main_content)
                
                # 提取返回值
                api_info["返回值"] = self.extract_detailed_return_value(main_content)
                
                # 提取调用示例
                api_info["调用示例"] = self.extract_detailed_example(main_content)
                
                # 提取约束限制
                api_info["约束限制"] = self.extract_constraints(main_content)
                
                # 提取相关接口
                api_info["相关接口"] = self.extract_related_apis(main_content)
                
                # 提取版本信息
                api_info["版本信息"] = self.extract_version_info(main_content)
            
            # 如果提取到了基本信息，返回API信息
            if any([api_info["API名称"], api_info["函数原型"], api_info["功能说明"]]):
                return api_info
                
        except Exception as e:
            logger.error(f"解析API页面失败 {url}: {str(e)}")
            with self.lock:
                self.failed_urls[url] = f"解析失败: {str(e)}"
            
        return None
    
    def extract_api_name_from_content(self, content) -> str:
        """从内容中提取API名称"""
        # 查找标题
        headings = content.find_all(['h1', 'h2', 'h3'])
        for heading in headings:
            text = self.clean_text(heading.get_text())
            if text and not re.search(r'(说明|描述|参数|返回|示例|约束)', text):
                # 清理标题，提取核心API名称
                api_name = re.sub(r'-.*?-.*', '', text)  # 移除类似 "-xxx-xxx" 的部分
                api_name = re.sub(r'（.*?）', '', api_name)  # 移除括号内容
                api_name = api_name.strip()
                if api_name:
                    return api_name
        
        return ""
    
    def extract_detailed_description(self, content) -> str:
        """提取详细功能说明"""
        descriptions = []
        
        # 查找功能说明相关的节
        desc_keywords = ['功能', '描述', '说明', '概述', '作用', 'function', 'description']
        
        for keyword in desc_keywords:
            # 查找包含关键词的标题
            headings = content.find_all(['h1', 'h2', 'h3', 'h4'], 
                                      string=re.compile(keyword, re.I))
            
            for heading in headings:
                # 获取标题后的内容
                desc_content = self.get_content_after_heading(heading)
                if desc_content:
                    descriptions.append(desc_content)
        
        # 如果没有找到特定的功能说明，查找第一个段落
        if not descriptions:
            paragraphs = content.find_all('p')
            for p in paragraphs[:3]:  # 检查前3个段落
                text = self.clean_text(p.get_text())
                if text and len(text) > 20:
                    descriptions.append(text)
                    break
        
        return ' '.join(descriptions)
    
    def extract_detailed_function_prototype(self, content) -> str:
        """提取详细函数原型"""
        prototypes = []
        
        # 查找代码块
        code_blocks = content.find_all(['code', 'pre'])
        
        for code in code_blocks:
            text = code.get_text().strip()
            
            # 查找函数声明模式
            func_patterns = [
                r'__aicore__.*?inline.*?\w+.*?\([^)]*\)',
                r'template.*?__aicore__.*?\w+.*?\([^)]*\)',
                r'\w+\s+\w+\s*\([^)]*\)\s*[;{]?',
                r'inline\s+\w+\s+\w+\s*\([^)]*\)'
            ]
            
            for pattern in func_patterns:
                matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    clean_match = self.clean_text(match)
                    if clean_match and len(clean_match) > 10:
                        prototypes.append(clean_match)
        
        # 去重并返回最长的原型
        if prototypes:
            return max(prototypes, key=len)
        
        return ""
    
    def extract_detailed_parameters(self, content) -> List[Dict]:
        """提取详细参数说明"""
        parameters = []
        
        # 查找参数表格
        tables = content.find_all('table')
        for table in tables:
            # 检查表格是否包含参数信息
            headers = table.find_all(['th'])
            header_text = ' '.join([h.get_text().lower() for h in headers])
            
            if re.search(r'(参数|parameter|param)', header_text):
                rows = table.find_all('tr')[1:]  # 跳过表头
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        param_info = {
                            "参数名": self.clean_text(cells[0].get_text()),
                            "类型": self.clean_text(cells[1].get_text()) if len(cells) > 1 else "",
                            "说明": self.clean_text(cells[2].get_text()) if len(cells) > 2 else ""
                        }
                        
                        # 如果只有两列，第二列可能是说明
                        if len(cells) == 2:
                            param_info["说明"] = param_info["类型"]
                            param_info["类型"] = ""
                        
                        if param_info["参数名"]:
                            parameters.append(param_info)
        
        # 查找参数列表
        if not parameters:
            param_sections = content.find_all(['div', 'section'], 
                                            string=re.compile(r'参数', re.I))
            
            for section in param_sections:
                parent = section.parent if section.parent else section
                lists = parent.find_all(['ul', 'ol', 'dl'])
                
                for plist in lists:
                    items = plist.find_all(['li', 'dt'])
                    for item in items:
                        text = self.clean_text(item.get_text())
                        # 解析参数格式
                        param_match = re.search(r'(\w+)\s*[：:-]\s*(.+)', text)
                        if param_match:
                            param_info = {
                                "参数名": param_match.group(1),
                                "类型": "",
                                "说明": param_match.group(2)
                            }
                            parameters.append(param_info)
        
        return parameters
    
    def extract_detailed_return_value(self, content) -> str:
        """提取详细返回值说明"""
        return_keywords = ['返回值', '返回', 'return', '输出']
        
        for keyword in return_keywords:
            # 查找返回值相关的标题
            headings = content.find_all(['h1', 'h2', 'h3', 'h4'], 
                                      string=re.compile(keyword, re.I))
            
            for heading in headings:
                return_content = self.get_content_after_heading(heading)
                if return_content:
                    return return_content
        
        # 查找返回值相关的表格
        tables = content.find_all('table')
        for table in tables:
            headers = table.find_all(['th'])
            header_text = ' '.join([h.get_text().lower() for h in headers])
            
            if re.search(r'(返回|return)', header_text):
                rows = table.find_all('tr')[1:]
                if rows:
                    cells = rows[0].find_all(['td', 'th'])
                    if cells:
                        return self.clean_text(cells[-1].get_text())
        
        return "无"
    
    def extract_detailed_example(self, content) -> str:
        """提取详细调用示例"""
        examples = []
        
        # 查找示例相关的标题
        example_keywords = ['示例', '例子', 'example', 'sample', '调用示例']
        
        for keyword in example_keywords:
            headings = content.find_all(['h1', 'h2', 'h3', 'h4'], 
                                      string=re.compile(keyword, re.I))
            
            for heading in headings:
                # 查找标题后的代码块
                next_element = heading
                for _ in range(10):  # 查找后续10个元素
                    next_element = next_element.find_next(['code', 'pre'])
                    if next_element:
                        code_text = self.clean_text(next_element.get_text())
                        if code_text and len(code_text) > 10:
                            examples.append(code_text)
                        break
        
        # 查找所有代码块中可能的示例
        if not examples:
            code_blocks = content.find_all(['code', 'pre'])
            for code in code_blocks:
                text = code.get_text().strip()
                # 如果代码块包含函数调用且较长，可能是示例
                if re.search(r'\w+\s*\([^)]*\)', text) and len(text) > 30:
                    examples.append(self.clean_text(text))
        
        return '\n'.join(examples) if examples else ""
    
    def extract_constraints(self, content) -> str:
        """提取约束限制信息"""
        constraint_keywords = ['约束', '限制', '注意', '要求', 'constraint', 'limitation', 'note']
        constraints = []
        
        for keyword in constraint_keywords:
            headings = content.find_all(['h1', 'h2', 'h3', 'h4'], 
                                      string=re.compile(keyword, re.I))
            
            for heading in headings:
                constraint_content = self.get_content_after_heading(heading)
                if constraint_content:
                    constraints.append(constraint_content)
        
        return ' '.join(constraints)
    
    def extract_related_apis(self, content) -> List[str]:
        """提取相关接口信息"""
        related = []
        
        # 查找相关接口链接
        links = content.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            text = link.get_text()
            
            if href and self.is_api_link(href, text):
                if text and text not in related:
                    related.append(text.strip())
        
        return related[:10]  # 限制数量
    
    def extract_version_info(self, content) -> str:
        """提取版本信息"""
        version_patterns = [
            r'版本\s*[:：]\s*([^\s]+)',
            r'Version\s*[:：]\s*([^\s]+)',
            r'v?\d+\.\d+(?:\.\d+)?',
            r'RC\d+',
            r'8\.2\.RC1'
        ]
        
        text = content.get_text()
        for pattern in version_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        
        return ""
    
    def get_content_after_heading(self, heading, max_elements: int = 5) -> str:
        """获取标题后的内容"""
        contents = []
        current = heading.next_sibling
        count = 0
        
        while current and count < max_elements:
            if hasattr(current, 'name'):
                # 如果遇到同级或更高级的标题，停止
                if current.name in ['h1', 'h2', 'h3', 'h4']:
                    break
                
                if current.name in ['p', 'div', 'span', 'li']:
                    text = self.clean_text(current.get_text())
                    if text and len(text) > 5:
                        contents.append(text)
                        count += 1
            elif isinstance(current, str):
                text = current.strip()
                if text and len(text) > 5:
                    contents.append(text)
                    count += 1
            
            current = current.next_sibling
        
        return ' '.join(contents)
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        # 移除HTML实体
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        return text
    
    def process_single_url(self, url: str) -> Optional[Dict]:
        """处理单个URL"""
        if url in self.visited_urls:
            return None
        
        with self.lock:
            self.visited_urls.add(url)
        
        soup = self.get_page_content(url)
        if soup:
            return self.extract_detailed_api_info(soup, url)
        
        return None
    
    def crawl_apis_parallel(self) -> Dict:
        """并行爬取API信息"""
        logger.info("开始并行爬取API信息...")
        
        # 获取主页面
        soup = self.get_page_content(self.base_url)
        if not soup:
            return self.generate_result()
        
        # 从主页面提取API信息
        main_api = self.extract_detailed_api_info(soup, self.base_url)
        if main_api:
            with self.lock:
                self.api_data.append(main_api)
        
        # 提取所有子页面链接
        all_links = self.extract_api_links(soup, self.base_url)
        logger.info(f"找到 {len(all_links)} 个API页面链接")
        
        # 并行处理所有链接
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_url = {executor.submit(self.process_single_url, url): url 
                           for url in all_links}
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    api_info = future.result()
                    if api_info:
                        with self.lock:
                            self.api_data.append(api_info)
                    
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"已完成 {completed}/{len(all_links)} 个页面")
                        
                except Exception as e:
                    logger.error(f"处理 {url} 时出错: {str(e)}")
                    with self.lock:
                        self.failed_urls[url] = f"处理异常: {str(e)}"
        
        logger.info("并行爬取完成")
        return self.generate_result()
    
    def generate_result(self) -> Dict:
        """生成最终结果"""
        # 去重
        seen_urls = set()
        unique_apis = []
        for api in self.api_data:
            if api["API文档URL"] not in seen_urls:
                seen_urls.add(api["API文档URL"])
                unique_apis.append(api)
        
        result = {
            "爬取时间": time.strftime('%Y-%m-%d %H:%M:%S'),
            "总计API数量": len(unique_apis),
            "成功爬取": len(unique_apis),
            "失败页面数": len(self.failed_urls),
            "爬取统计": {
                "有函数原型": len([api for api in unique_apis if api["函数原型"]]),
                "有参数说明": len([api for api in unique_apis if api["参数说明"]]),
                "有调用示例": len([api for api in unique_apis if api["调用示例"]]),
                "有功能说明": len([api for api in unique_apis if api["功能说明"]])
            },
            "APIs": unique_apis,
            "错误信息": self.failed_urls
        }
        
        return result

def main():
    """主函数"""
    base_url = "https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0003.html"
    
    scraper = EnhancedAscendAPIScraper(base_url, max_workers=8)
    result = scraper.crawl_apis_parallel()
    
    # 输出结果到文件
    output_file = "/workspace/enhanced_ascend_apis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"增强版爬取完成！结果已保存到: {output_file}")
    logger.info(f"总计API数量: {result['总计API数量']}")
    logger.info(f"失败页面数: {result['失败页面数']}")
    logger.info(f"爬取统计: {result['爬取统计']}")
    
    # 输出结果摘要
    print(json.dumps({
        "爬取摘要": {
            "爬取时间": result["爬取时间"],
            "总计API数量": result["总计API数量"],
            "成功爬取": result["成功爬取"],
            "失败页面数": result["失败页面数"],
            "爬取统计": result["爬取统计"]
        },
        "前5个API示例": result["APIs"][:5],
        "错误信息": result["错误信息"]
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()