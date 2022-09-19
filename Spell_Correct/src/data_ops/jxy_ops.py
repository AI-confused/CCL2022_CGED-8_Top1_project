import os
import sys
from tqdm import tqdm
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)
import random
random.seed(0)
from src.helps.help_data import *
class ZengQiang(object):

    def __init__(self, err_path):
        self.errbase = jsonload(err_path)
        self.xz = [i for i in range(100)]
        random.shuffle(self.xz)
        random.shuffle(self.xz)
        self.spans = [([0,50], 'err'), ([50, 85], 'yj'), ([85,100],'xj')]
        self.s2t = {}
        for x in self.spans:
            for i in range(x[0][0], x[0][1]):
                self.s2t[i] = x[1]

    def parser(self, sInput):
        nlen = len(sInput)
        sInput2 = list(sInput)
        ch_index = [] # 中文字符索引
        for i,x in enumerate(sInput2):
           if is_chinese(x):
            ch_index.append(i)
        # todo 高于5个字选1个错别字，每长十个字再长1个
        numb = max(int((nlen-5)/10) + 1, 0)
        if ch_index:
            add_err_index = random.sample(ch_index, numb)
        else:
            add_err_index = []
        for index in add_err_index:
            char = sInput[index]
            new_char = self.get_err_char(char)
            sInput2[index] = new_char
        return "".join(sInput2)

    def get_err_char(self, schar):
        if schar not in self.errbase:
            return schar
        stype = self.s2t[random.sample(self.xz, 1)[0]]
        if stype in self.errbase[schar] and self.errbase[schar][stype]:
            return random.sample(self.errbase[schar][stype], 1)[0]
        return schar


def get_all_data(datapath):
    ''' 获取lang8 '''
    rfile = openrfile(datapath)
    hou = {}
    for i,line in tqdm(enumerate(rfile)):
        temp = strQ2B(line).lower().strip()
        temp = re.sub(' ', '', temp).split('\t')
        hou.setdefault(temp[1], 0)
    return hou

def zegnqiang_v0(config):
    ''' 文本增强 第一个版本  '''
    tool = ZengQiang(config['word_base'])
    lang8_second = get_all_data(config['lang8_path'])
    sens = list(lang8_second.keys())
    random.shuffle(sens)
    random.shuffle(sens)
    wfile = openwfile(os.path.join(config['data_save_path'], "jxy_dev.jsonl"))
    for i, x in tqdm(enumerate(sens[:1000])):
        y = tool.parser(x)
        error = []
        for j,k in enumerate(x):
            if k != y[j]:
                error.append({"start": j, "end": j+2, "type": "S", "answer": x[j]})
        onedatas = {"id":str(i), 'from':'lang8.增强.v0.dev', 'text':y, 'correction':x, "error": error}
        wfile.write(json.dumps(onedatas, ensure_ascii=False) + "\n")
    wfile.close()

    wfile = openwfile(os.path.join(config['data_save_path'], "jxy_train.jsonl"))
    for i,x in tqdm(enumerate(sens[1000:])):
        y = tool.parser(x)
        error = []
        for j,k in enumerate(x):
            if k != y[j]:
                error.append({"start": j, "end": j+2, "type": "S", "answer": x[j]})
        onedatas = {"id":str(i), 'from':'lang8.增强.v0.train', 'text':y, 'correction':x, "error": error}
        wfile.write(json.dumps(onedatas, ensure_ascii=False) + "\n")
    wfile.close()

if __name__ == "__main__":
    # config = {
    #     'word_base': 易错字库,
    #     'lang8_path': lang8数据,
    #      'data_save_path': 数据存放地址
    # }
    config = {
        'word_base': r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/代码提交内容/datas/错别字库_v1.json",
        'lang8_path': r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas/track1/lang8.train.ccl22.para",
         'data_save_path': r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/代码提交内容/datas"
    }
    zegnqiang_v0(config)
   