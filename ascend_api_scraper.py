#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华为昇腾API文档爬虫
爬取指定页面及其子页面的API信息，按照指定格式输出JSON结果
"""

import requests
import json
import re
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Set
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AscendAPIScaper:
    def __init__(self, base_url: str):
        self.base_url = base_url
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
            self.failed_urls[url] = error_msg
            return None
    
    def extract_api_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """提取API相关链接"""
        links = []
        
        # 查找所有包含API信息的链接
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
                
            # 构建完整URL
            full_url = urljoin(base_url, href)
            
            # 过滤API相关链接
            if self.is_api_link(href, link.get_text()):
                links.append(full_url)
                
        # 查找目录结构中的API链接
        nav_elements = soup.find_all(['nav', 'div'], class_=re.compile(r'(nav|menu|toc|sidebar)', re.I))
        for nav in nav_elements:
            for link in nav.find_all('a', href=True):
                href = link.get('href')
                if href and self.is_api_link(href, link.get_text()):
                    full_url = urljoin(base_url, href)
                    links.append(full_url)
        
        return list(set(links))  # 去重
    
    def is_api_link(self, href: str, text: str) -> bool:
        """判断是否为API相关链接"""
        # API相关的URL模式
        api_patterns = [
            r'api.*\.html',
            r'ascendcopapi',
            r'atlasascendc_api',
            r'function',
            r'interface'
        ]
        
        # API相关的文本模式
        text_patterns = [
            r'API',
            r'接口',
            r'函数',
            r'方法',
            r'算子',
            r'操作符'
        ]
        
        href_lower = href.lower()
        text_lower = text.lower() if text else ''
        
        # 检查URL模式
        for pattern in api_patterns:
            if re.search(pattern, href_lower):
                return True
                
        # 检查文本模式
        for pattern in text_patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False
    
    def extract_api_info(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """从页面中提取API信息"""
        apis = []
        
        try:
            # 方法1: 查找API标题和内容块
            api_sections = self.find_api_sections(soup)
            
            for section in api_sections:
                api_info = self.parse_api_section(section, url)
                if api_info:
                    apis.append(api_info)
            
            # 方法2: 如果没有找到标准的API节，尝试解析整个页面
            if not apis:
                api_info = self.parse_full_page_api(soup, url)
                if api_info:
                    apis.append(api_info)
                    
        except Exception as e:
            error_msg = f"解析API信息失败: {str(e)}"
            logger.error(f"{url} - {error_msg}")
            self.failed_urls[url] = error_msg
            
        return apis
    
    def find_api_sections(self, soup: BeautifulSoup) -> List:
        """查找API节"""
        sections = []
        
        # 查找包含API信息的节
        # 方法1: 通过标题查找
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'], 
                                string=re.compile(r'(API|接口|函数|方法)', re.I))
        
        for heading in headings:
            # 获取该标题下的内容
            section = self.get_section_content(heading)
            if section:
                sections.append(section)
        
        # 方法2: 通过class或id查找
        api_containers = soup.find_all(['div', 'section'], 
                                     class_=re.compile(r'(api|function|interface)', re.I))
        sections.extend(api_containers)
        
        return sections
    
    def get_section_content(self, heading) -> Optional:
        """获取标题下的内容节"""
        content = []
        current = heading.next_sibling
        
        while current:
            if hasattr(current, 'name'):
                # 如果遇到同级或更高级的标题，停止
                if current.name in ['h1', 'h2', 'h3', 'h4']:
                    break
                content.append(current)
            current = current.next_sibling
            
        return content if content else None
    
    def parse_api_section(self, section, url: str) -> Optional[Dict]:
        """解析API节内容"""
        try:
            api_info = {
                "API名称": "",
                "API文档URL": url,
                "功能说明": "",
                "函数原型": "",
                "参数说明": [],
                "返回值": "",
                "调用示例": ""
            }
            
            # 如果section是列表，转换为BeautifulSoup对象
            if isinstance(section, list):
                section_html = ''.join(str(item) for item in section)
                section_soup = BeautifulSoup(section_html, 'html.parser')
            else:
                section_soup = section
            
            # 提取API名称
            api_name = self.extract_api_name(section_soup)
            if api_name:
                api_info["API名称"] = api_name
            
            # 提取功能说明
            description = self.extract_description(section_soup)
            if description:
                api_info["功能说明"] = description
            
            # 提取函数原型
            prototype = self.extract_function_prototype(section_soup)
            if prototype:
                api_info["函数原型"] = prototype
            
            # 提取参数说明
            parameters = self.extract_parameters(section_soup)
            if parameters:
                api_info["参数说明"] = parameters
            
            # 提取返回值
            return_value = self.extract_return_value(section_soup)
            if return_value:
                api_info["返回值"] = return_value
            
            # 提取调用示例
            example = self.extract_example(section_soup)
            if example:
                api_info["调用示例"] = example
            
            # 如果提取到了基本信息，返回API信息
            if api_info["API名称"] or api_info["函数原型"]:
                return api_info
                
        except Exception as e:
            logger.error(f"解析API节失败: {str(e)}")
            
        return None
    
    def parse_full_page_api(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """解析整个页面的API信息"""
        try:
            api_info = {
                "API名称": "",
                "API文档URL": url,
                "功能说明": "",
                "函数原型": "",
                "参数说明": [],
                "返回值": "",
                "调用示例": ""
            }
            
            # 从页面标题或主要内容中提取API名称
            title = soup.find('title')
            if title:
                api_info["API名称"] = self.clean_text(title.get_text())
            
            # 查找主要内容区域
            main_content = (soup.find('main') or 
                          soup.find('div', class_=re.compile(r'content', re.I)) or
                          soup.find('div', class_=re.compile(r'main', re.I)) or
                          soup.find('body'))
            
            if main_content:
                # 提取各种信息
                api_info["功能说明"] = self.extract_description(main_content)
                api_info["函数原型"] = self.extract_function_prototype(main_content)
                api_info["参数说明"] = self.extract_parameters(main_content)
                api_info["返回值"] = self.extract_return_value(main_content)
                api_info["调用示例"] = self.extract_example(main_content)
            
            # 如果提取到了基本信息，返回API信息
            if any([api_info["API名称"], api_info["函数原型"], api_info["功能说明"]]):
                return api_info
                
        except Exception as e:
            logger.error(f"解析整页API失败: {str(e)}")
            
        return None
    
    def extract_api_name(self, soup) -> str:
        """提取API名称"""
        # 查找标题中的API名称
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        for heading in headings:
            text = self.clean_text(heading.get_text())
            if text and not re.search(r'(说明|描述|参数|返回|示例)', text):
                return text
        
        # 查找代码块中的函数名
        code_blocks = soup.find_all(['code', 'pre'])
        for code in code_blocks:
            text = code.get_text()
            # 查找函数定义模式
            func_match = re.search(r'(\w+)\s*\(', text)
            if func_match:
                return func_match.group(1)
        
        return ""
    
    def extract_description(self, soup) -> str:
        """提取功能说明"""
        # 查找描述相关的文本
        desc_patterns = [
            r'(功能|描述|说明|作用)',
            r'(function|description|purpose)',
        ]
        
        for pattern in desc_patterns:
            # 查找包含描述关键词的元素
            desc_elements = soup.find_all(text=re.compile(pattern, re.I))
            for element in desc_elements:
                parent = element.parent
                if parent:
                    # 获取该元素后面的文本
                    next_text = self.get_following_text(parent)
                    if next_text and len(next_text) > 10:
                        return self.clean_text(next_text)
        
        # 如果没有找到特定的描述，尝试获取第一段文本
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = self.clean_text(p.get_text())
            if text and len(text) > 10:
                return text
        
        return ""
    
    def extract_function_prototype(self, soup) -> str:
        """提取函数原型"""
        # 查找代码块
        code_blocks = soup.find_all(['code', 'pre'])
        
        for code in code_blocks:
            text = code.get_text().strip()
            # 查找函数声明模式
            if re.search(r'\w+\s*\([^)]*\)\s*[;{]?', text):
                # 清理代码块，提取函数声明
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if re.search(r'^\s*\w+.*\w+\s*\([^)]*\)', line):
                        return self.clean_text(line)
        
        return ""
    
    def extract_parameters(self, soup) -> List[Dict]:
        """提取参数说明"""
        parameters = []
        
        # 查找参数表格
        tables = soup.find_all('table')
        for table in tables:
            # 检查表格是否包含参数信息
            headers = table.find_all(['th', 'td'])
            header_text = ' '.join([h.get_text().lower() for h in headers[:5]])
            
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
                        if param_info["参数名"]:
                            parameters.append(param_info)
        
        # 如果没有找到表格，查找列表形式的参数
        if not parameters:
            param_lists = soup.find_all(['ul', 'ol', 'dl'])
            for plist in param_lists:
                items = plist.find_all(['li', 'dt', 'dd'])
                for item in items:
                    text = self.clean_text(item.get_text())
                    # 解析参数格式: 参数名 - 说明
                    param_match = re.search(r'(\w+)\s*[-:：]\s*(.+)', text)
                    if param_match:
                        param_info = {
                            "参数名": param_match.group(1),
                            "类型": "",
                            "说明": param_match.group(2)
                        }
                        parameters.append(param_info)
        
        return parameters
    
    def extract_return_value(self, soup) -> str:
        """提取返回值"""
        # 查找返回值相关的文本
        return_patterns = [
            r'(返回值|返回|return)',
        ]
        
        for pattern in return_patterns:
            elements = soup.find_all(text=re.compile(pattern, re.I))
            for element in elements:
                parent = element.parent
                if parent:
                    next_text = self.get_following_text(parent)
                    if next_text:
                        return self.clean_text(next_text)
        
        return "无"
    
    def extract_example(self, soup) -> str:
        """提取调用示例"""
        # 查找示例代码块
        example_keywords = ['示例', '例子', 'example', 'sample']
        
        for keyword in example_keywords:
            # 查找包含示例关键词的元素
            example_elements = soup.find_all(text=re.compile(keyword, re.I))
            for element in example_elements:
                parent = element.parent
                if parent:
                    # 查找后续的代码块
                    next_element = parent.find_next(['code', 'pre'])
                    if next_element:
                        return self.clean_text(next_element.get_text())
        
        # 如果没有找到特定的示例，查找所有代码块
        code_blocks = soup.find_all(['code', 'pre'])
        for code in code_blocks:
            text = code.get_text().strip()
            # 如果代码块包含函数调用，可能是示例
            if re.search(r'\w+\s*\([^)]*\)\s*;?', text) and len(text) > 20:
                return self.clean_text(text)
        
        return ""
    
    def get_following_text(self, element) -> str:
        """获取元素后面的文本"""
        texts = []
        current = element.next_sibling
        
        count = 0
        while current and count < 3:  # 限制查找范围
            if hasattr(current, 'get_text'):
                text = current.get_text().strip()
                if text:
                    texts.append(text)
            elif isinstance(current, str):
                text = current.strip()
                if text:
                    texts.append(text)
            current = current.next_sibling
            count += 1
        
        return ' '.join(texts)
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        # 移除特殊字符
        text = re.sub(r'[^\w\s\-_().,;:：，。；]', '', text)
        
        return text
    
    def crawl_apis(self) -> Dict:
        """爬取API信息"""
        logger.info("开始爬取API信息...")
        
        # 获取主页面
        soup = self.get_page_content(self.base_url)
        if not soup:
            return self.generate_result()
        
        self.visited_urls.add(self.base_url)
        
        # 从主页面提取API信息
        apis = self.extract_api_info(soup, self.base_url)
        self.api_data.extend(apis)
        
        # 提取子页面链接
        sub_links = self.extract_api_links(soup, self.base_url)
        logger.info(f"找到 {len(sub_links)} 个子页面链接")
        
        # 爬取子页面
        for link in sub_links:
            if link not in self.visited_urls:
                self.visited_urls.add(link)
                time.sleep(1)  # 避免请求过快
                
                sub_soup = self.get_page_content(link)
                if sub_soup:
                    sub_apis = self.extract_api_info(sub_soup, link)
                    self.api_data.extend(sub_apis)
        
        return self.generate_result()
    
    def generate_result(self) -> Dict:
        """生成最终结果"""
        result = {
            "总计API数量": len(self.api_data),
            "成功爬取": len(self.api_data),
            "失败页面数": len(self.failed_urls),
            "APIs": self.api_data,
            "错误信息": self.failed_urls
        }
        
        return result

def main():
    """主函数"""
    base_url = "https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0003.html"
    
    scraper = AscendAPIScaper(base_url)
    result = scraper.crawl_apis()
    
    # 输出结果到文件
    output_file = "/workspace/ascend_apis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"爬取完成！结果已保存到: {output_file}")
    logger.info(f"总计API数量: {result['总计API数量']}")
    logger.info(f"失败页面数: {result['失败页面数']}")
    
    # 输出结果到控制台
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()