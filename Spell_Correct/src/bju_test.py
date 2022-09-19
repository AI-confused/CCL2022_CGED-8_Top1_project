import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
import torch
from transformers import BertTokenizer
from munch import DefaultMunch
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from src.helps.help_data import *
from src.baseline.model import BERT_Model as USE_MODEL

class Decoder:
    def __init__(self, config):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.model = USE_MODEL(config)
        self.model.to(self.device)
        self.config = config
        self.batch = config.batch_size
        self.max_seq_len = config.max_seq_len
        vocab_path = os.path.join(config['pretrained_model'], 'vocab.txt')
        self.vocab = []
        with open(vocab_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            self.vocab.append(line.strip())

    def init(self):
        self.model.load_state_dict(torch.load(self.config.model_path))
    
    def parser(self, inputs):
        # todo 处理数据
        data = []
        nlens = []
        for src_line in inputs:
            data_sample = {}
            src_token_list = list(src_line)
            nlens.append(min(len(src_line), self.max_seq_len-2))
            src_token_list.insert(0, '[CLS]')
            src_token_list.append('[SEP]')
            if len(src_token_list) < self.max_seq_len:
                src_token_list += ['[PAN]'] * (self.max_seq_len - len(src_token_list))
            else:
                src_token_list = src_token_list[:self.max_seq_len]
                src_token_list[-1] = '[SEP]'

            data_sample['src_text'] = src_line
            data_sample['input_ids'] = self.tokenizer.convert_tokens_to_ids(src_token_list)
            data_sample['token_type_ids'] = [0 for i in range(len(src_token_list))]
            data_sample['attention_mask'] = [1 for i in range(len(src_token_list))]
            data.append(data_sample)

        # 正式推理
        values = []
        indices = []
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(inputs), self.batch):
                end = min(len(inputs), start+self.batch)
                data_temp = data[start: end]
                batch_temp = {
                    'input_ids': [x['input_ids'] for x in data_temp],
                    'token_type_ids': [x['token_type_ids'] for x in data_temp],
                    'attention_mask': [x['attention_mask'] for x in data_temp]
                }
                batch = {k: torch.tensor(v).to(self.device) for k, v in batch_temp.items()}
                _, logits = self.model(**batch)
                output = torch.argmax(logits, dim=-1)
                value = logits.topk(k=5,dim=2,largest=True, sorted=True).values
                indice = logits.topk(k=5,dim=2,largest=True, sorted=True).indices
                value = torch.nn.Softmax(dim=-1)(value)
                values += value.cpu().numpy().tolist()
                indices += indice.cpu().numpy().tolist()
                outputs += output.cpu().numpy().tolist()
        # todo 转换格式
        relist = []
        for i in range(len(nlens)):
            leni = nlens[i]
            reondata = {
                "text": inputs[i],
                'correction': "",
                'optype': [],
                'optype_p': [],
                'char_p':[]
            }
            char_p = []
            correction = []
            # 循环每个位置  增加逻辑
            for nid in range(1, leni + 1):
                charlist = [] # 存储每个字 以及它的概率
                for j in range(len(indices[i][nid])):
                    charlist.append((self.vocab[indices[i][nid][j]], format(values[i][nid][j], '.4f')))
                char_p.append(charlist)
                y = reondata['text'][nid-1]
                # todo 如果预测的 和 原句 都是中文字符 才会预测是错别字
                if is_chinese(y) and is_chinese(charlist[0][0]):
                        if y == charlist[0][0]: # 生成同字符
                            correction.append(y)
                        else:
                            get_char_s = [c[0] for c in charlist]
                            if y not in get_char_s: # 原字符不在 top5
                                correction.append(charlist[0][0])
                            else:
                                index_c = get_char_s.index(y)
                                top1 = charlist[0][1]
                                topy = charlist[index_c][1]
                                pdiv = float(topy)/float(top1)
                                if pdiv < 0.01:
                                    correction.append(charlist[0][0])
                                else:
                                    correction.append(y)
                else: # 否则 保持 原字符
                    correction.append(y)
            reondata['char_p'] = char_p
            reondata["correction"] = "".join(correction)
            if len(reondata['correction']) < len(reondata['text']):
                reondata['correction'] += reondata['text'][len(reondata['correction']):]
            answer = []
            s = -1
            for nid,x in enumerate(reondata["text"]):
                if nid >= self.max_seq_len - 2:
                    continue
                if x == reondata["correction"][nid]:
                    if s != -1:
                        answer.append([s, nid, reondata["correction"][s:nid]])
                        s = -1
                    reondata["optype"].append("K")
                    reondata['optype_p'].append(char_p[nid][0][1])
                else:
                    reondata["optype"].append("S")
                    if s == -1:
                        s = nid
                    reondata['optype_p'].append(char_p[nid][0][1])
            reondata['answer'] = answer
            # if answer:
            #     print(answer)
            relist.append(reondata)
        return relist
    
    def reload_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

if __name__ == "__main__":

    # config = {
    #     'pretrained_model': r"预训练模型地址",
    #     'model_path': r"训练好的参数地址", 
    #     'test_path': r"测试数据地址",
    #     'save_path': r"测试数据预测存储地址",
    #     'max_seq_len': 128,
    #     'batch_size': 48,
    #     'label_ignore_id': 0,
    #     'dropout': 0.1
    # }

    config = {
        'pretrained_model': r"/home/hufei/pyapp/miaoxie_xiuci/auto_text_classifier/atc/data/bert_base_chinese",
        'model_path': r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/代码提交内容/models/bju/lang8-epoch-100.pt", 
        'test_path': r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/代码提交内容/datas/cged_test.txt",
        'save_path': r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/代码提交内容/datas/out_病句S.txt",
        'max_seq_len': 128,
        'batch_size': 48,
        'label_ignore_id': 0,
        'dropout': 0.1
    }

    args = DefaultMunch.fromDict(config)
    decoder = Decoder(args)
    decoder.init()

    getdata = decoder.parser(["明天天天气怎么养？"])
    for x in getdata:
        print(x)
        print()

    # todo 语病 测试集
    rfile = openrfile(config['test_path'])
    wfile = openwfile(config['save_path'])
    ids = []
    inputs = []
    for line in rfile:
        temp = line.strip().split('\t')
        ids.append(temp[0])
        inputs.append(temp[1])
    getdata = decoder.parser(inputs)
    for i,x in enumerate(getdata):
        text = x['text']
        correction = x['correction']
        if text != correction:
            s = -1
            for j,y in enumerate(text):
                if y != correction[j]:
                    if s == -1:
                        s = j + 1
                else:
                    if s != -1:
                        temp = [ids[i], str(s), str(j), 'S', correction[s-1:j]]
                        wfile.write(", ".join(temp)+'\n')
                        s = -1
            if s != -1:
                temp = [ids[i], str(s), str(len(text)), 'S', correction[s-1:]]
                wfile.write(", ".join(temp)+'\n')
