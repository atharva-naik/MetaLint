class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        dst_s = 0
        re_num1 = re.compile('^\s*-{0,1}\d+')  # '  -234' 
        re_num2 = re.compile('^\s*\+{0,1}\d+')  # '  +234'
        num_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        is_neg = False
        match1 = re_num1.search(str)  # 不论字符串中有几组数字，只取第一组
        match2 = re_num2.search(str)  # 针对+123的情况
        if match1:
            num_str = match1.group()
            num_str = num_str.strip()
            if num_str.startswith('-'):
                is_neg = True
                num_str = num_str[1:]
            for i in range(0, len(num_str)):
                dst_s = dst_s * 10 + num_dict.get(num_str[i], 0)
        elif match2:
            num_str = match2.group()
            num_str = num_str.strip()
            if num_str.startswith('+'):
                num_str = num_str[1:]
            for i in range(0, len(num_str)):
                dst_s = dst_s * 10 + num_dict.get(num_str[i], 0)
        dst_s = -dst_s if is_neg else dst_s
        # 判断溢出
        threshold = pow(2,31)
        if dst_s >= threshold:
            dst_s = threshold -1
        elif dst_s < -threshold:
            dst_s = -threshold
        return dst_s