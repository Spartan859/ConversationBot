"""
数字转换模块
将阿拉伯数字转换为中文数字，用于ASR后处理
"""

import re


class NumberConverter:
    """阿拉伯数字转中文数字转换器"""
    
    # 数字映射
    DIGIT_MAP = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    
    # 单位映射
    UNIT_MAP = ['', '十', '百', '千']
    BIG_UNIT_MAP = ['', '万', '亿']
    
    @classmethod
    def convert_year(cls, year_str: str) -> str:
        """
        转换年份：2013 → 二零一三
        """
        return ''.join(cls.DIGIT_MAP[d] for d in year_str)
    
    @classmethod
    def convert_number(cls, num_str: str) -> str:
        """
        转换普通数字：137 → 一百三十七
        处理 0-9999 范围内的数字
        """
        num = int(num_str)
        
        if num == 0:
            return '零'
        
        if num < 10:
            return cls.DIGIT_MAP[str(num)]
        
        if num < 20:
            if num == 10:
                return '十'
            return '十' + cls.DIGIT_MAP[str(num % 10)]
        
        result = []
        
        # 处理千位
        if num >= 1000:
            result.append(cls.DIGIT_MAP[str(num // 1000)])
            result.append('千')
            num %= 1000
            if num < 100 and num > 0:
                result.append('零')
        
        # 处理百位
        if num >= 100:
            result.append(cls.DIGIT_MAP[str(num // 100)])
            result.append('百')
            num %= 100
            if num < 10 and num > 0:
                result.append('零')
        
        # 处理十位
        if num >= 10:
            if num >= 20:  # 20以上需要加数字
                result.append(cls.DIGIT_MAP[str(num // 10)])
            result.append('十')
            num %= 10
        
        # 处理个位
        if num > 0:
            result.append(cls.DIGIT_MAP[str(num)])
        
        return ''.join(result)
    
    @classmethod
    def convert_decimal(cls, decimal_str: str) -> str:
        """
        转换小数：29.84 → 二十九点八四
        """
        parts = decimal_str.split('.')
        integer_part = cls.convert_number(parts[0]) if parts[0] != '0' else '零'
        
        if len(parts) == 2:
            decimal_part = ''.join(cls.DIGIT_MAP[d] for d in parts[1])
            return f"{integer_part}点{decimal_part}"
        
        return integer_part
    
    @classmethod
    def convert_percentage(cls, match) -> str:
        """
        转换百分比：46% → 百分之四十六
        """
        num_str = match.group(1)
        
        # 处理小数百分比
        if '.' in num_str:
            num_part = cls.convert_decimal(num_str)
        else:
            num_part = cls.convert_number(num_str)
        
        return f"百分之{num_part}"
    
    @classmethod
    def convert_date(cls, match) -> str:
        """
        转换日期：10月22日 → 十月二十二日
        """
        month = match.group(1)
        day = match.group(2)
        
        month_cn = cls.convert_number(month)
        day_cn = cls.convert_number(day)
        
        return f"{month_cn}月{day_cn}日"
    
    @classmethod
    def convert_year_full(cls, match) -> str:
        """
        转换完整年份：2013年 → 二零一三年
        """
        year = match.group(1)
        year_cn = cls.convert_year(year)
        return f"{year_cn}年"
    
    @classmethod
    def convert_year_range(cls, match) -> str:
        """
        转换年份范围：2015-2016 → 二零一五至二零一六
        """
        year1 = match.group(1)
        year2 = match.group(2)
        
        year1_cn = cls.convert_year(year1)
        year2_cn = cls.convert_year(year2)
        
        return f"{year1_cn}至{year2_cn}"
    
    @classmethod
    def convert_standalone_number(cls, match) -> str:
        """
        转换独立数字（带单位）：137人 → 一百三十七人
        """
        num_str = match.group(1)
        unit = match.group(2)
        
        # 处理小数
        if '.' in num_str:
            num_cn = cls.convert_decimal(num_str)
        else:
            num = int(num_str)
            if num < 10000:
                num_cn = cls.convert_number(num_str)
            else:
                # 大数字保持阿拉伯数字（简化处理）
                return match.group(0)
        
        return f"{num_cn}{unit}"
    
    @classmethod
    def convert_text(cls, text: str) -> str:
        """
        转换文本中的所有阿拉伯数字为中文数字
        
        Args:
            text: 输入文本
            
        Returns:
            转换后的文本
        """
        # 1. 转换年份范围（必须在单独年份之前）
        text = re.sub(r'(\d{4})-(\d{4})', cls.convert_year_range, text)
        
        # 2. 转换百分比
        text = re.sub(r'(\d+(?:\.\d+)?)%', cls.convert_percentage, text)
        
        # 3. 转换日期
        text = re.sub(r'(\d{1,2})月(\d{1,2})日', cls.convert_date, text)
        
        # 4. 转换完整年份
        text = re.sub(r'(\d{4})年', cls.convert_year_full, text)
        
        # 5. 转换带单位的数字（人、米、个、次、名等）
        # 常见单位
        units = ['人', '米', '个', '次', '名', '元', '万', '亿', '岁', '时', '分', '秒', '天', '月', '年']
        for unit in units:
            text = re.sub(rf'(\d+(?:\.\d+)?)({unit})', cls.convert_standalone_number, text)
        
        # 6. 转换剩余的独立数字（谨慎处理，避免误转换）
        # 只转换较短的数字（1-4位）
        def convert_remaining(match):
            num_str = match.group(0)
            if len(num_str) <= 4:
                return cls.convert_number(num_str)
            return num_str
        
        # 转换被空格或标点包围的独立数字
        text = re.sub(r'(?<=[\s，。、；：！？,\.;:!?\(\)\[\]（）【】])\d{1,4}(?=[\s，。、；：！？,\.;:!?\(\)\[\]（）【】])', 
                      convert_remaining, text)
        
        return text


# 测试代码
if __name__ == "__main__":
    converter = NumberConverter()
    
    test_cases = [
        "特别是70年代或80年代初。",
        "四川男女排召开了2015-2016赛季新闻发布会。",
        "46%正在考虑移民国外。",
        "新华网南京10月22日消息。",
        "被贴条人数137人。",
        "叫2014年全面增长60%。",
        "深圳为16.5%。",
        "2010年11月4日。",
        "享年96岁。",
        "出生于1986年6月16日的帕万。",
        "相比2013年多出26个。",
        "王正在已经取得近130米领先优势的大好局面下。",
        "2013年年实现净利润29.84万元。",
    ]
    
    print("=" * 80)
    print("数字转换测试")
    print("=" * 80)
    
    for text in test_cases:
        converted = converter.convert_text(text)
        if text != converted:
            print(f"原文: {text}")
            print(f"转换: {converted}")
            print("-" * 80)
