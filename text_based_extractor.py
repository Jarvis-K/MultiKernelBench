#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于文本内容的华为昇腾API信息提取器
直接从获取到的页面文本中提取API信息，避免HTML结构解析错误
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

class TextBasedAPIExtractor:
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
        
    def get_page_text(self, url: str) -> Optional[str]:
        """获取页面的纯文本内容"""
        try:
            logger.info(f"正在访问: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 获取纯文本
            text = soup.get_text()
            
            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            error_msg = f"访问页面失败: {str(e)}"
            logger.error(f"{url} - {error_msg}")
            with self.lock:
                self.failed_urls[url] = error_msg
            return None
    
    def extract_api_links_from_text(self, text: str, base_url: str) -> List[str]:
        """从文本中提取API链接"""
        links = set()
        
        # 查找所有的API相关URL
        url_patterns = [
            r'https://www\.hiascend\.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/[^"\s]+\.html',
            r'/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/[^"\s]+\.html'
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if not match.startswith('http'):
                    match = urljoin(base_url, match)
                links.add(match)
        
        return list(links)
    
    def extract_api_name_from_text(self, text: str, url: str) -> str:
        """从文本中提取API名称"""
        # 方法1: 从URL中提取
        url_match = re.search(r'atlasascendc_api_07_(\d+)\.html', url)
        if url_match:
            api_id = url_match.group(1)
        
        # 方法2: 查找标题模式
        title_patterns = [
            r'([A-Za-z][A-Za-z0-9_]*)\s*-.*?-.*?API',
            r'([A-Za-z][A-Za-z0-9_]*)\s*-.*?指令',
            r'([A-Za-z][A-Za-z0-9_]*)\s*-.*?函数',
            r'([A-Za-z][A-Za-z0-9_]*)\s*-.*?操作',
            r'([A-Za-z][A-Za-z0-9_]*)\s*接口',
            r'接口名称[：:]\s*([A-Za-z][A-Za-z0-9_]*)',
            r'函数名[：:]\s*([A-Za-z][A-Za-z0-9_]*)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                api_name = match.group(1)
                # 过滤掉一些通用词
                if api_name not in ['API', 'CANN', 'Ascend', 'C', 'RC1']:
                    return api_name
        
        # 方法3: 从文档标题中提取
        title_match = re.search(r'^([^-\s]+)', text.strip())
        if title_match:
            potential_name = title_match.group(1)
            if re.match(r'^[A-Za-z][A-Za-z0-9_]*$', potential_name):
                return potential_name
        
        return f"API_{api_id}" if 'api_id' in locals() else "Unknown"
    
    def extract_function_description(self, text: str) -> str:
        """从文本中提取功能说明"""
        # 查找功能说明相关的段落
        desc_patterns = [
            r'功能说明[：:]\s*([^。]+。)',
            r'功能描述[：:]\s*([^。]+。)',
            r'接口功能[：:]\s*([^。]+。)',
            r'说明[：:]\s*([^。]+。)',
            r'用途[：:]\s*([^。]+。)',
            r'作用[：:]\s*([^。]+。)',
            r'本接口用于\s*([^。]+。)',
            r'该接口\s*([^。]+。)',
            r'此函数\s*([^。]+。)',
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, text)
            if match:
                desc = match.group(1).strip()
                if len(desc) > 10 and not re.search(r'(参数|返回|示例)', desc):
                    return desc
        
        # 如果没有找到特定的功能说明，尝试提取第一段有意义的描述
        sentences = re.split(r'[。！？\n]', text)
        for sentence in sentences[:10]:  # 只检查前10句
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                not re.search(r'(版权|Copyright|昇腾|CANN|文档|目录)', sentence) and
                re.search(r'[计算处理操作执行实现支持提供]', sentence)):
                return sentence + '。'
        
        return ""
    
    def extract_function_prototype(self, text: str) -> str:
        """从文本中提取函数原型"""
        # 查找函数声明模式
        prototype_patterns = [
            r'(__aicore__\s+inline\s+[^{;]+\([^)]*\))',
            r'(template\s*<[^>]*>\s*__aicore__\s+inline\s+[^{;]+\([^)]*\))',
            r'(inline\s+\w+\s+\w+\s*\([^)]*\))',
            r'(\w+\s+\w+\s*\([^)]*\)\s*;)',
            r'函数原型[：:]\s*([^\n]+)',
            r'接口原型[：:]\s*([^\n]+)',
            r'声明[：:]\s*([^\n]+)',
        ]
        
        for pattern in prototype_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                prototype = match.strip()
                # 验证是否是有效的函数原型
                if (len(prototype) > 10 and 
                    '(' in prototype and ')' in prototype and
                    not re.search(r'[。！？]', prototype)):
                    # 清理原型
                    prototype = re.sub(r'\s+', ' ', prototype)
                    return prototype
        
        return ""
    
    def extract_parameters(self, text: str) -> List[Dict]:
        """从文本中提取参数说明"""
        parameters = []
        
        # 查找参数表格或列表
        param_section_patterns = [
            r'参数说明.*?(?=返回值|示例|约束|$)',
            r'输入参数.*?(?=输出参数|返回值|示例|约束|$)',
            r'参数列表.*?(?=返回值|示例|约束|$)',
            r'参数.*?(?=返回值|示例|约束|$)',
        ]
        
        param_section = ""
        for pattern in param_section_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                param_section = match.group(0)
                break
        
        if param_section:
            # 提取参数信息
            param_patterns = [
                r'(\w+)\s+([^：:]+)[：:]\s*([^。\n]+)',
                r'(\w+)[：:]\s*([^。\n]+)',
                r'•\s*(\w+)\s*[：:-]\s*([^。\n]+)',
                r'-\s*(\w+)\s*[：:-]\s*([^。\n]+)',
            ]
            
            for pattern in param_patterns:
                matches = re.findall(pattern, param_section)
                for match in matches:
                    if len(match) == 3:
                        param_name, param_type, param_desc = match
                    else:
                        param_name, param_desc = match
                        param_type = ""
                    
                    # 清理参数信息
                    param_name = param_name.strip()
                    param_type = param_type.strip()
                    param_desc = param_desc.strip()
                    
                    if param_name and len(param_name) < 50:  # 避免提取到错误的内容
                        parameters.append({
                            "参数名": param_name,
                            "类型": param_type,
                            "说明": param_desc
                        })
        
        return parameters[:10]  # 限制参数数量
    
    def extract_return_value(self, text: str) -> str:
        """从文本中提取返回值"""
        return_patterns = [
            r'返回值[：:]\s*([^。\n]+)',
            r'返回[：:]\s*([^。\n]+)',
            r'输出[：:]\s*([^。\n]+)',
            r'return\s*[：:]\s*([^。\n]+)',
        ]
        
        for pattern in return_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return_val = match.group(1).strip()
                if return_val and len(return_val) < 200:
                    return return_val
        
        # 检查是否是void函数
        if re.search(r'void\s+\w+\s*\(', text):
            return "无"
        
        return "无"
    
    def extract_example(self, text: str) -> str:
        """从文本中提取调用示例"""
        example_patterns = [
            r'示例[：:]?\s*([^。]*(?:template|__aicore__|inline)[^}]*}?)',
            r'调用示例[：:]?\s*([^。]*(?:template|__aicore__|inline)[^}]*}?)',
            r'使用示例[：:]?\s*([^。]*(?:template|__aicore__|inline)[^}]*}?)',
            r'代码示例[：:]?\s*([^。]*(?:template|__aicore__|inline)[^}]*}?)',
            r'(template\s*<[^>]*>\s*[^{;]+\([^)]*\))',
            r'(__aicore__\s+inline\s+[^{;]+\([^)]*\))',
        ]
        
        for pattern in example_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                example = match.strip()
                if (len(example) > 20 and 
                    ('(' in example and ')' in example) and
                    not re.search(r'[。！？]', example[:50])):  # 前50个字符不应该有句号
                    # 清理示例代码
                    example = re.sub(r'\s+', ' ', example)
                    return example
        
        return ""
    
    def extract_api_info_from_text(self, text: str, url: str) -> Optional[Dict]:
        """从文本中提取完整的API信息"""
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
            
            # 提取各个字段
            api_info["API名称"] = self.extract_api_name_from_text(text, url)
            api_info["功能说明"] = self.extract_function_description(text)
            api_info["函数原型"] = self.extract_function_prototype(text)
            api_info["参数说明"] = self.extract_parameters(text)
            api_info["返回值"] = self.extract_return_value(text)
            api_info["调用示例"] = self.extract_example(text)
            
            return api_info
            
        except Exception as e:
            logger.error(f"从文本提取API信息失败 {url}: {str(e)}")
            with self.lock:
                self.failed_urls[url] = f"文本解析失败: {str(e)}"
            
        return None
    
    def process_single_url(self, url: str) -> Optional[Dict]:
        """处理单个URL"""
        if url in self.visited_urls:
            return None
        
        with self.lock:
            self.visited_urls.add(url)
        
        text = self.get_page_text(url)
        if text:
            return self.extract_api_info_from_text(text, url)
        
        return None
    
    def crawl_apis_from_text(self) -> Dict:
        """基于文本内容爬取API信息"""
        logger.info("开始基于文本的API信息爬取...")
        
        # 获取主页面文本
        main_text = self.get_page_text(self.base_url)
        if not main_text:
            return self.generate_result()
        
        # 从主页面提取API信息
        main_api = self.extract_api_info_from_text(main_text, self.base_url)
        if main_api:
            with self.lock:
                self.api_data.append(main_api)
        
        # 从主页面文本中提取所有API链接
        all_links = self.extract_api_links_from_text(main_text, self.base_url)
        logger.info(f"从文本中提取到 {len(all_links)} 个API页面链接")
        
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
        
        logger.info("基于文本的爬取完成")
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
            "爬取方式": "基于文本内容提取",
            "总计API数量": len(unique_apis),
            "成功爬取": len(unique_apis),
            "失败页面数": len(self.failed_urls),
            "数据质量统计": {
                "有API名称": len([api for api in unique_apis if api["API名称"]]),
                "有功能说明": len([api for api in unique_apis if api["功能说明"]]),
                "有函数原型": len([api for api in unique_apis if api["函数原型"]]),
                "有参数说明": len([api for api in unique_apis if api["参数说明"]]),
                "有返回值说明": len([api for api in unique_apis if api["返回值"] and api["返回值"] != "无"]),
                "有调用示例": len([api for api in unique_apis if api["调用示例"]])
            },
            "APIs": unique_apis,
            "错误信息": self.failed_urls
        }
        
        return result

def main():
    """主函数"""
    base_url = "https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0003.html"
    
    extractor = TextBasedAPIExtractor(base_url, max_workers=6)
    result = extractor.crawl_apis_from_text()
    
    # 输出结果到文件
    output_file = "/workspace/text_based_ascend_apis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"基于文本的爬取完成！结果已保存到: {output_file}")
    logger.info(f"总计API数量: {result['总计API数量']}")
    logger.info(f"失败页面数: {result['失败页面数']}")
    logger.info(f"数据质量统计: {result['数据质量统计']}")
    
    # 输出结果摘要
    print(json.dumps({
        "爬取摘要": {
            "爬取时间": result["爬取时间"],
            "爬取方式": result["爬取方式"],
            "总计API数量": result["总计API数量"],
            "成功爬取": result["成功爬取"],
            "失败页面数": result["失败页面数"],
            "数据质量统计": result["数据质量统计"]
        },
        "前5个API示例": result["APIs"][:5] if result["APIs"] else [],
        "错误信息数量": len(result["错误信息"])
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()