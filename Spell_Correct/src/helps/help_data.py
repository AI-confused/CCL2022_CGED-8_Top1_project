#-*-coding:utf-8-*-
"""
@date: 2021. 02. 06
@auther: Hu Fei
@doc: 帮助文档，方便平时开发
"""

import re
import os
import json


def get_json_path(path):
    '''
    获取当前文件下的所有json路径
    :param path:
    :return:
    '''
    list = []
    filelist = os.listdir(path)
    for file in filelist:
        if file.endswith(".json"):
            list.append(os.path.join(path, file))
        elif os.path.isdir(os.path.join(path, file)):
            addlist = get_json_path(os.path.join(path, file))
            list += addlist
    return list

def get_txt_path(path):
    '''
    获取当前文件下的所有txt路径
    :param path:
    :return:
    '''
    list = []
    filelist = os.listdir(path)
    for file in filelist:
        if file.endswith(".txt"):
            list.append(os.path.join(path, file))
        elif os.path.isdir(os.path.join(path, file)):
            addlist = get_txt_path(os.path.join(path, file))
            list += addlist
    return list

def is_chinese(char):
    '''
    如果是中文字符，则返回 True
    '''
    pattern_num_comma = r"[\u4E00-\u9FA5]"
    return re.match(pattern_num_comma, char)

def findkey(sString):
    """
    返回字符串中的中文字符串
    :param sString:
    :return:
    """
    key = ""
    for x in sString:
        if is_chinese(x):
            key += x
    return key

def openrfile(filename):
    return open(filename, "r", encoding="utf-8")

def openwfile(filename):
    return open(filename, "w", encoding="utf-8")

def jsondump(data,filepath):
    json.dump(data, openwfile(filepath), ensure_ascii=False, indent=1)

def jsonload(filepath):
    return json.load(openrfile(filepath))

def calculate_pcf(bz, pr, hit):
    pre = 0
    cal = 0
    f1 = 0
    if bz != 0:
        cal = hit * 1.0 / bz

    if pr != 0:
        pre = hit * 1.0 / pr

    if cal + pre != 0:
        f1 = 2 * cal * pre / ( cal + pre )

    return pre,cal,f1

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

