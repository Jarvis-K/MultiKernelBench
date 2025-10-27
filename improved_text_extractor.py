#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的基于文本内容的华为昇腾API信息提取器
结合已有的URL列表，对每个页面进行精确的文本分析提取
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

class ImprovedTextAPIExtractor:
    def __init__(self, max_workers: int = 6):
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
        self.failed_urls: Dict[str, str] = {}
        self.api_data: List[Dict] = []
        self.lock = threading.Lock()
        
        # 从之前的结果中加载URL列表
        self.load_existing_urls()
        
    def load_existing_urls(self):
        """从之前的爬取结果中加载URL列表"""
        self.api_urls = []
        try:
            # 尝试从增强版结果中加载
            with open('/workspace/enhanced_ascend_apis.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                for api in data.get('APIs', []):
                    url = api.get('API文档URL')
                    if url:
                        self.api_urls.append(url)
            
            logger.info(f"从已有结果中加载了 {len(self.api_urls)} 个API URL")
            
        except Exception as e:
            logger.warning(f"无法加载已有URL列表: {e}")
            # 如果无法加载，使用一些常见的API URL作为示例
            self.api_urls = [
                "https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0003.html"
            ]
    
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
            
            # 清理文本，保留换行符用于段落识别
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
            
            return '\n'.join(lines)
            
        except Exception as e:
            error_msg = f"访问页面失败: {str(e)}"
            logger.error(f"{url} - {error_msg}")
            with self.lock:
                self.failed_urls[url] = error_msg
            return None
    
    def extract_api_name_from_text(self, text: str, url: str) -> str:
        """从文本中提取API名称"""
        lines = text.split('\n')
        
        # 方法1: 从页面标题中提取
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            # 查找标题模式
            title_patterns = [
                r'^([A-Za-z][A-Za-z0-9_]*)\s*-',  # API名称-其他信息
                r'^([A-Za-z][A-Za-z0-9_]*)\s*（',  # API名称（说明）
                r'^([A-Za-z][A-Za-z0-9_]*)\s*$',   # 单独的API名称
            ]
            
            for pattern in title_patterns:
                match = re.match(pattern, line)
                if match:
                    api_name = match.group(1)
                    # 过滤掉一些通用词
                    if api_name not in ['API', 'CANN', 'Ascend', 'C', 'RC1', 'document', 'detail']:
                        return api_name
        
        # 方法2: 从URL中提取API ID
        url_match = re.search(r'atlasascendc_api_07_(\d+)\.html', url)
        if url_match:
            api_id = url_match.group(1)
            
            # 尝试在文本中查找对应的函数名
            func_patterns = [
                r'([A-Za-z][A-Za-z0-9_]*)\s*\(',  # 函数调用
                r'接口名称[：:]\s*([A-Za-z][A-Za-z0-9_]*)',
                r'函数名[：:]\s*([A-Za-z][A-Za-z0-9_]*)',
            ]
            
            for pattern in func_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if (len(match) > 2 and 
                        match not in ['API', 'CANN', 'Ascend', 'C', 'RC1', 'void', 'int', 'float']):
                        return match
            
            return f"API_{api_id}"
        
        return "Unknown"
    
    def extract_function_description(self, text: str) -> str:
        """从文本中提取功能说明"""
        lines = text.split('\n')
        descriptions = []
        
        # 查找功能说明相关的行
        desc_keywords = ['功能说明', '功能描述', '接口功能', '用途', '作用', '功能']
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # 检查是否包含功能说明关键词
            for keyword in desc_keywords:
                if keyword in line:
                    # 获取当前行和后续几行作为功能说明
                    desc_lines = []
                    
                    # 如果当前行有内容，提取冒号后的部分
                    if '：' in line or ':' in line:
                        desc_part = re.split('[：:]', line, 1)
                        if len(desc_part) > 1 and desc_part[1].strip():
                            desc_lines.append(desc_part[1].strip())
                    
                    # 获取后续相关行
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if (next_line and 
                            len(next_line) > 10 and
                            not re.match(r'(参数|返回|示例|约束|表\d+)', next_line)):
                            desc_lines.append(next_line)
                        elif next_line and re.match(r'(参数|返回|示例)', next_line):
                            break
                    
                    if desc_lines:
                        desc = ' '.join(desc_lines)
                        if len(desc) > 15:
                            descriptions.append(desc)
                        break
        
        # 如果没有找到特定的功能说明，查找描述性段落
        if not descriptions:
            for i, line in enumerate(lines):
                line = line.strip()
                if (len(line) > 30 and 
                    not re.match(r'(版权|Copyright|昇腾|CANN|文档|目录|表\d+)', line) and
                    re.search(r'[计算处理操作执行实现支持提供用于实现]', line) and
                    '。' in line):
                    # 取第一个句子
                    sentences = re.split('[。！？]', line)
                    if sentences and len(sentences[0]) > 20:
                        descriptions.append(sentences[0] + '。')
                        break
        
        return descriptions[0] if descriptions else ""
    
    def extract_function_prototype(self, text: str) -> str:
        """从文本中提取函数原型"""
        lines = text.split('\n')
        prototypes = []
        
        # 查找函数原型相关的行
        proto_keywords = ['函数原型', '接口原型', '声明', '原型']
        
        for i, line in enumerate(lines):
            # 方法1: 查找标记的函数原型
            for keyword in proto_keywords:
                if keyword in line and ('：' in line or ':' in line):
                    proto_part = re.split('[：:]', line, 1)
                    if len(proto_part) > 1:
                        prototype = proto_part[1].strip()
                        if prototype and len(prototype) > 10:
                            prototypes.append(prototype)
                    
                    # 检查后续行
                    for j in range(i + 1, min(i + 3, len(lines))):
                        next_line = lines[j].strip()
                        if (next_line and 
                            '(' in next_line and ')' in next_line and
                            len(next_line) > 10):
                            prototypes.append(next_line)
                            break
        
        # 方法2: 直接查找函数声明模式
        prototype_patterns = [
            r'(__aicore__\s+inline\s+[^{;]+\([^)]*\))',
            r'(template\s*<[^>]*>\s*__aicore__[^{;]+\([^)]*\))',
            r'(inline\s+\w+\s+\w+\s*\([^)]*\))',
            r'(\w+\s+\w+\s*\([^)]*\)\s*;?)',
        ]
        
        for pattern in prototype_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                prototype = match.strip()
                # 验证是否是有效的函数原型
                if (len(prototype) > 15 and 
                    '(' in prototype and ')' in prototype and
                    not re.search(r'[。！？]', prototype)):
                    prototypes.append(prototype)
        
        # 返回最合适的原型
        if prototypes:
            # 优先返回包含__aicore__的原型
            for proto in prototypes:
                if '__aicore__' in proto:
                    return re.sub(r'\s+', ' ', proto)
            
            # 否则返回最长的原型
            return re.sub(r'\s+', ' ', max(prototypes, key=len))
        
        return ""
    
    def extract_parameters(self, text: str) -> List[Dict]:
        """从文本中提取参数说明"""
        lines = text.split('\n')
        parameters = []
        
        # 查找参数说明段落
        param_start = -1
        param_end = -1
        
        for i, line in enumerate(lines):
            if re.search(r'参数说明|输入参数|参数列表|参数', line):
                param_start = i
                break
        
        if param_start >= 0:
            # 找到参数段落的结束
            for i in range(param_start + 1, len(lines)):
                if re.search(r'返回值|示例|约束|调用示例', lines[i]):
                    param_end = i
                    break
            
            if param_end == -1:
                param_end = min(param_start + 20, len(lines))
            
            # 提取参数段落
            param_lines = lines[param_start:param_end]
            param_text = '\n'.join(param_lines)
            
            # 解析参数信息
            param_patterns = [
                r'(\w+)\s*[：:]\s*([^。\n]+)',  # 参数名：说明
                r'(\w+)\s+([^：:\n]+)[：:]\s*([^。\n]+)',  # 参数名 类型：说明
                r'•\s*(\w+)[：:-]\s*([^。\n]+)',  # • 参数名：说明
                r'-\s*(\w+)[：:-]\s*([^。\n]+)',  # - 参数名：说明
            ]
            
            for pattern in param_patterns:
                matches = re.findall(pattern, param_text)
                for match in matches:
                    if len(match) == 2:
                        param_name, param_desc = match
                        param_type = ""
                    else:
                        param_name, param_type, param_desc = match
                    
                    # 清理和验证参数信息
                    param_name = param_name.strip()
                    param_type = param_type.strip() if param_type else ""
                    param_desc = param_desc.strip()
                    
                    # 过滤掉无效的参数
                    if (param_name and 
                        len(param_name) < 50 and 
                        len(param_name) > 1 and
                        not re.search(r'[。！？]', param_name) and
                        param_desc):
                        
                        parameters.append({
                            "参数名": param_name,
                            "类型": param_type,
                            "说明": param_desc
                        })
                
                if parameters:
                    break
        
        return parameters[:10]  # 限制参数数量
    
    def extract_return_value(self, text: str) -> str:
        """从文本中提取返回值"""
        lines = text.split('\n')
        
        # 查找返回值相关的行
        return_keywords = ['返回值', '返回', '输出', 'return']
        
        for i, line in enumerate(lines):
            for keyword in return_keywords:
                if keyword in line and ('：' in line or ':' in line):
                    return_part = re.split('[：:]', line, 1)
                    if len(return_part) > 1:
                        return_val = return_part[1].strip()
                        if return_val and len(return_val) < 100:
                            return return_val
                    
                    # 检查后续行
                    for j in range(i + 1, min(i + 3, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and len(next_line) < 100 and not re.search(r'(参数|示例)', next_line):
                            return next_line
        
        # 检查是否是void函数
        if re.search(r'void\s+\w+\s*\(', text):
            return "无"
        
        return "无"
    
    def extract_example(self, text: str) -> str:
        """从文本中提取调用示例"""
        lines = text.split('\n')
        examples = []
        
        # 查找示例相关的行
        example_keywords = ['示例', '调用示例', '使用示例', '代码示例', 'example']
        
        for i, line in enumerate(lines):
            for keyword in example_keywords:
                if keyword in line:
                    # 获取后续的代码行
                    example_lines = []
                    for j in range(i + 1, min(i + 10, len(lines))):
                        next_line = lines[j].strip()
                        if (next_line and 
                            ('(' in next_line and ')' in next_line) or
                            'template' in next_line or
                            '__aicore__' in next_line):
                            example_lines.append(next_line)
                        elif next_line and re.search(r'[。！？]', next_line[:20]):
                            break
                    
                    if example_lines:
                        example = ' '.join(example_lines)
                        if len(example) > 20:
                            examples.append(example)
                        break
        
        # 如果没有找到标记的示例，查找代码模式
        if not examples:
            code_patterns = [
                r'(template\s*<[^>]*>\s*[^{;]+\([^)]*\))',
                r'(__aicore__\s+inline\s+[^{;]+\([^)]*\))',
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) > 30:
                        examples.append(match)
        
        if examples:
            # 清理并返回最好的示例
            best_example = max(examples, key=len)
            return re.sub(r'\s+', ' ', best_example)
        
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
        text = self.get_page_text(url)
        if text:
            return self.extract_api_info_from_text(text, url)
        return None
    
    def crawl_all_apis(self) -> Dict:
        """爬取所有API信息"""
        logger.info(f"开始改进的文本提取，处理 {len(self.api_urls)} 个API页面...")
        
        # 并行处理所有链接
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_url = {executor.submit(self.process_single_url, url): url 
                           for url in self.api_urls}
            
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
                    if completed % 20 == 0:
                        logger.info(f"已完成 {completed}/{len(self.api_urls)} 个页面")
                        
                except Exception as e:
                    logger.error(f"处理 {url} 时出错: {str(e)}")
                    with self.lock:
                        self.failed_urls[url] = f"处理异常: {str(e)}"
        
        logger.info("改进的文本提取完成")
        return self.generate_result()
    
    def generate_result(self) -> Dict:
        """生成最终结果"""
        result = {
            "爬取时间": time.strftime('%Y-%m-%d %H:%M:%S'),
            "爬取方式": "改进的基于文本内容提取",
            "总计API数量": len(self.api_data),
            "成功爬取": len(self.api_data),
            "失败页面数": len(self.failed_urls),
            "数据质量统计": {
                "有API名称": len([api for api in self.api_data if api["API名称"] and api["API名称"] != "Unknown"]),
                "有功能说明": len([api for api in self.api_data if api["功能说明"]]),
                "有函数原型": len([api for api in self.api_data if api["函数原型"]]),
                "有参数说明": len([api for api in self.api_data if api["参数说明"]]),
                "有返回值说明": len([api for api in self.api_data if api["返回值"] and api["返回值"] != "无"]),
                "有调用示例": len([api for api in self.api_data if api["调用示例"]])
            },
            "APIs": self.api_data,
            "错误信息": self.failed_urls
        }
        
        return result

def main():
    """主函数"""
    extractor = ImprovedTextAPIExtractor(max_workers=8)
    result = extractor.crawl_all_apis()
    
    # 输出结果到文件
    output_file = "/workspace/improved_text_based_apis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"改进的文本提取完成！结果已保存到: {output_file}")
    logger.info(f"总计API数量: {result['总计API数量']}")
    logger.info(f"失败页面数: {result['失败页面数']}")
    logger.info(f"数据质量统计: {result['数据质量统计']}")
    
    # 输出结果摘要和示例
    print(json.dumps({
        "爬取摘要": {
            "爬取时间": result["爬取时间"],
            "爬取方式": result["爬取方式"],
            "总计API数量": result["总计API数量"],
            "成功爬取": result["成功爬取"],
            "失败页面数": result["失败页面数"],
            "数据质量统计": result["数据质量统计"]
        },
        "前3个完整API示例": result["APIs"][:3] if result["APIs"] else [],
        "有完整信息的API数量": len([api for api in result["APIs"] 
                                if api["API名称"] != "Unknown" and 
                                   api["功能说明"] and 
                                   api["函数原型"]]),
        "错误信息数量": len(result["错误信息"])
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()