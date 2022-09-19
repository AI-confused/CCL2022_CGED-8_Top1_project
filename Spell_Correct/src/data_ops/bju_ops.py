import os,sys,json
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)
from src.helps.help_data import *
from src.helps.zh_wiki import zh2Hans
from src.helps.langconfig import traditional_to_simple

import random
random.seed(0)
import Levenshtein
mask = "[unused1]"

def get_all_data(config):
    ''' 获取lang8 '''
    datapath = config['lang8_path']
    redatas = []
    rfile = openrfile(datapath)
    qian = {}
    hou = {}
    for i,line in tqdm(enumerate(rfile)):
        temp = strQ2B(line).lower().strip()
        temp = re.sub(' ', '', temp).split('\t')
        qian.setdefault(temp[0], 0)
        hou.setdefault(temp[1], 0)
        if len(temp) != 2:
            print(line)
        else:
            redatas.append(temp)
    print("数据总量：", i + 1)
    print("前字符串去重：", len(qian))
    print("后字符串去重：", len(hou))
    # jsondump(hou, r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas/get_data_or/lang8.second.json")

    return redatas

def get_top1_answer(lang8_data):
    '''
    获取 编辑距离 长度绝对值 随机选一个
    '''
    data2bz = {}
    for i,temp in tqdm(enumerate(lang8_data)):

        # 1: 两边相差不能超过 5
        if abs(len(temp[0]) - len(temp[1])) > 5:
            continue
        # 2：中文字符 占比 大于 0.5
        elif not count_chinese_p(temp[0]):
            continue
        
        # 3: 含有 繁体
        elif is_fanti(temp[0]) or is_fanti(temp[1]):
            # print(temp)
            temp2 = [traditional_to_simple(temp[0]), traditional_to_simple(temp[1])]
        
        else:
            temp2 = temp
        
        data2bz.setdefault(temp2[0], [])
        if temp2[1] not in data2bz[temp2[0]]:
            data2bz[temp2[0]].append(temp2[1])
    
    up2numb = 0
    eq2numb = 0
    print('第一字符串个数：', len(data2bz))
    for x in data2bz:
        if len(data2bz[x]) > 1:
            data2bz[x] = get_top1(x, data2bz[x])
        
        if len(data2bz[x]) == 2:
            eq2numb += 1
        elif len(data2bz[x]) > 2:
            up2numb += 1
        
        if len(data2bz[x]) > 1:
            data2bz[x] = random.sample(data2bz[x], 1)
        
        data2bz[x] = data2bz[x][0]
    # jsondump(data2bz, r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas/get_data_or/lang8.oneanswer.json")
    print("1numb:", len(data2bz) - up2numb - eq2numb, "2numb:", eq2numb, "up2numb:", up2numb)
    return data2bz

def count_chinese_p(strtemp, p=0.5):
    nlen = len(strtemp)
    n = 0
    for x in strtemp:
        if is_chinese(x):
            n += 1
    if n/nlen > p:
        return True
    else:
        return False

def get_top1(skey, values):
    temp = values
    # todo 先，编辑距离相近 后，长度选择最相近的
    nlen = [(Levenshtein.distance(skey, x), abs(len(skey) - len(x)), i) for i,x in enumerate(values)]
    nlen.sort()

    nlen2 = []
    nlen3 = []
    if nlen[0][0] < nlen[1][0]: # 选最接近的
        return [values[nlen[0][2]]]
    else:
        # 相差绝对值
        nlen2 = [(x[1], x[2]) for x in nlen if x[0] == nlen[0][0]]
        nlen2.sort()

        if nlen2[0][0] < nlen2[1][0]:
            return [values[nlen2[0][1]]]
        else:
            nlen3 = [x for x in nlen2 if x[0] == nlen2[0][0]]
            temp = [values[x[1]] for x in nlen3]

    return temp

def is_fanti(strtemp):
    nlen = 0
    n = 0
    for x in strtemp:
        if is_chinese(x):
            nlen += 1
        if x in zh2Hans:
            n += 1
    
    if n  and nlen - n < int(4/5*nlen):
        return True
    else:
        return False

def pair2edits_new(src, trg):
    src_line = ''.join(src.strip().replace(',', '，').split())
    tgt_line = ''.join(trg.strip().replace(',', '，').split())
    # sid = lines_sid[i].strip().split('\t')[0]

    # edits = Levenshtein.opcodes(src_line, tgt_line)
    # reverse
    _edits = Levenshtein.opcodes(src_line[::-1], tgt_line[::-1])[::-1]
    edits = []
    src_len = len(src_line)
    tgt_len = len(tgt_line)
    for edit in _edits:
        edits.append((edit[0], src_len - edit[2], src_len - edit[1], tgt_len - edit[4], tgt_len - edit[3]))

    # merge coterminous Levenshtein edited spans
    merged_edits = []
    for edit in edits:
        if edit[0] == 'equal':
            continue
        if len(merged_edits) > 0:
            last_edit = merged_edits[-1]
            if last_edit[0] == 'insert' and edit[0] == 'insert' and last_edit[2] == edit[1]:
                new_edit = ('insert', last_edit[1], edit[2], last_edit[3], edit[4])
                merged_edits[-1] = new_edit
            elif last_edit[2] == edit[1]:
                assert last_edit[4] == edit[3]
                new_edit = ('hybrid', last_edit[1], edit[2], last_edit[3], edit[4])
                merged_edits[-1] = new_edit
            elif last_edit[0] == 'insert' and edit[0] == 'delete' \
                and tgt_line[last_edit[3]:last_edit[4]] == src_line[edit[1]:edit[2]]:
                new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                merged_edits[-1] = new_edit
            elif last_edit[0] == 'delete' and edit[0] == 'insert':
                if src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]]:
                # print(src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]])
                    new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                    merged_edits[-1] = new_edit
                elif edit[4] < len(tgt_line) and tgt_line[edit[3]] == tgt_line[edit[4]] and src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]+1:edit[4]+1]:
                    new_edit = ('luanxu', last_edit[1], edit[2]+1, last_edit[3], edit[4])
                    merged_edits[-1] = new_edit
                else:
                    merged_edits.append(edit)
            else:
                merged_edits.append(edit)
        else:
            merged_edits.append(edit)
    merged_edits2 = []
    for edit in merged_edits:
        if edit[0] == 'equal':
            continue
        if len(merged_edits2) > 0:
            last_edit = merged_edits2[-1]
            if last_edit[0] == 'insert' and edit[0] == 'insert' and last_edit[2] == edit[1]:
                new_edit = ('insert', last_edit[1], edit[2], last_edit[3], edit[4])
                merged_edits2[-1] = new_edit
            elif last_edit[2] == edit[1]:
                assert last_edit[4] == edit[3]
                new_edit = ('hybrid', last_edit[1], edit[2], last_edit[3], edit[4])
                merged_edits2[-1] = new_edit
            elif last_edit[0] == 'insert' and edit[0] == 'delete' \
                and tgt_line[last_edit[3]:last_edit[4]] == src_line[edit[1]:edit[2]]:
                new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                merged_edits2[-1] = new_edit
            elif last_edit[0] == 'delete' and edit[0] == 'insert':
                if src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]]:
                # print(src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]])
                    new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                    merged_edits2[-1] = new_edit
                elif edit[4] < len(tgt_line) and tgt_line[edit[3]] == tgt_line[edit[4]] and src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]+1:edit[4]+1]:
                    new_edit = ('luanxu', last_edit[1], edit[2]+1, last_edit[3], edit[4])
                    merged_edits2[-1] = new_edit
                else:
                    merged_edits2.append(edit)
            else:
                merged_edits2.append(edit)
        else:
            merged_edits2.append(edit)
    # generate edit sequence
    result = []
    for edit in merged_edits2:
        if tgt_line[edit[3]:edit[4]] == '[UNK]':
            continue
        if edit[0] == "insert":
            result.append((edit[1]+1, edit[1]+1, "M", tgt_line[edit[3]:edit[4]]))
        elif edit[0] == "replace":
            # new_op = post_process_S(src_line, (str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
            result.append((edit[1]+1, edit[2], "S", tgt_line[edit[3]:edit[4]]))
        elif edit[0] == "delete":
            result.append((edit[1]+1, edit[2], "R"))
        elif edit[0] == "hybrid":
            # new_op = post_process_S(src_line, (str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
            result.append((edit[1]+1, edit[2], "S", tgt_line[edit[3]:edit[4]]))
        elif edit[0] == "luanxu":
            result.append((edit[1]+1, edit[2] , "W", tgt_line[edit[3]:edit[4]]))

    # print
    # out_line = id +',\t'
    return result

def to_char_pattern(lang8_one_answer_data):
    redatas = []
    # print(lang8_one_answer_data)
    for i,x in enumerate(lang8_one_answer_data):
        result = pair2edits_new(x, lang8_one_answer_data[x])
        result.sort()
        redata = {"id": i, "from":'lang8', "text": x, "correct": lang8_one_answer_data[x], "error": []}
        for x in result:
            if x[0] > x[1]:
                pass
            if x[2] == 'S':
                redata["error"].append({"start": x[0]-1, "end": x[1], "type": 'S', "answer": x[3]})
            elif x[2] == 'M':
                redata["error"].append({"start": x[0]-1, "end": x[1], "type": 'M', "answer": x[3]})
            elif x[2] == 'W':
                redata["error"].append({"start": x[0]-1, "end": x[1], "type": 'W'})
            elif x[2] == 'R':
                redata["error"].append({"start": x[0]-1, "end": x[1], "type": 'R'})
        redatas.append(redata)
    return redatas

def one_yangben(onedata):
    text = onedata['text']
    error = onedata['error']
    correct = list(text)
    text = list(text)
    for e in error:
        if e['type'] == "R":
            for i in range(e['start'], e['end']):
                correct[i] = mask
        elif e['type'] == "S":
            if e['end'] - e['start'] >= len(e['answer']):
                for i in range(e['start'], e['start']+len(e['answer'])):
                    correct[i] = e['answer'][i-e['start']]
                for i in range(e['start']+len(e['answer']), e['end']):
                    correct[i] = mask
            else:
                for i in range(e['start'], e['end']):
                    correct[i] = e['answer'][i-e['start']]

    onedata['text'] = text
    onedata['correct'] = correct

    # print(text)
    # print(correct)

    return onedata
    
def save2train(char_pattern_data, save_path):
    datas = char_pattern_data
    wfile = openwfile(save_path)
    for onedata in tqdm(datas):
        get_data = one_yangben(onedata)
        wfile.write(json.dumps(get_data, ensure_ascii=False) + "\n")
    wfile.close()


if __name__ == "__main__":

    # todo 处理 lang8 数据, 每个正确字符串对应一种错误方式
    # 仅保留其中的 S 和 R 类型
    # R类型对应位置直接用 [unused1] 代替
    # S 修正后长于原串内容，直接截取，短于原串内容用 [unused1] 代替
    config = {
        "lang8_path1": r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas/track1/lang8.train.ccl22.para", # 举办方提供的lang8数据
        "lang8_path": r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/代码提交内容/datas/lang8.train.ccl22.para", # 举办方提供的lang8数据
        "train_data_save_path": r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/代码提交内容/datas/bju_train.jsonl" # 训练集存储格式
    }
    lang8_data = get_all_data(config)
    lang8_one_answer_data = get_top1_answer(lang8_data)
    char_pattern_data = to_char_pattern(lang8_one_answer_data)
    save2train(char_pattern_data, config["train_data_save_path"])
    
    ''' 验证集是使用 2020年测试集 处理成相同格式 已给出 '''

