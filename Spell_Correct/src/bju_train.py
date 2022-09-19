import os,sys,json
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from tqdm import tqdm
import torch
from random import seed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from transformers import BertTokenizer, AdamW, get_scheduler
from tokenizers import normalizers
from src.helps.help_data import jsondump, calculate_pcf
from src.baseline.model import BERT_Model
from src.baseline.eval_char_level import get_char_metrics
from src.baseline.eval_sent_level import get_sent_metrics

class Padding_in_batch:

    def __init__(self, input_pad_id):
        self.input_pad_id = input_pad_id

    def pad(self, seq, pad_id, max_len):
        pad_part = torch.LongTensor([pad_id] * (max_len - seq.shape[-1]))
        pad_seq = torch.cat([seq, pad_part], dim=-1)
        return pad_seq

    def __call__(self, batch):
        max_len = 0
        for item in batch:
            max_len = max(max_len, len(item["input_ids"]))

        if "trg_ids" in item:
            is_test = False
        else:
            is_test = True

        for item in batch:
            item['input_ids'] = self.pad(item['input_ids'], self.input_pad_id, max_len)
            item['token_type_ids'] = self.pad(item['token_type_ids'], 0, max_len)
            item['attention_mask'] = self.pad(item['attention_mask'], 0, max_len)
            if not is_test:
                item['trg_ids'] = self.pad(item['trg_ids'], self.input_pad_id, max_len)

        return default_collate(batch)

class CBZ_Dataset(Dataset):

    def __init__(self, path, config, subset, tokenizer):
        self.path = path
        self.config = config
        self.max_seq_len = config['max_seq_len']
        self.subset = subset
        self.tokenizer = tokenizer
        self.data = self.read_data()

    def read_data(self):
        # 从文件中读取数据
        parallel_data = self.get_data_from_jsonl()
        # 调整数据格式
        data = self.encode_parallel_data(self.config, parallel_data)
        # data = data[:10000]
        return data
    
    def get_data_from_jsonl(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
        parallel_data = []
        c_no_error_sent = 0 # 没有错别字的句子
        for line in lines:
            onedata = json.loads(line)
            id = onedata["id"]
            src_sent = onedata["text"]
            trg_sent = onedata["correct"]

            modification = []
            if src_sent == trg_sent:
                c_no_error_sent += 1
            else:
                for i, (src_char, trg_char) in enumerate(zip(src_sent, trg_sent)):
                    if src_char != trg_char:
                        modification.append((i, trg_char))
            # print(id, src_sent, trg_sent, modification)
            parallel_data.append((id, src_sent, trg_sent, modification))
        print("--------------------------------")
        print(self.subset + ": " + str(len(lines)))
        print("error-free sentences: " + str(c_no_error_sent))
        print("--------------------------------\n")
        return parallel_data

    def encode_parallel_data(self, config, parallel_data):
        data = []
        normalizer = normalizers.Sequence([normalizers.Lowercase()])
        for item in tqdm(parallel_data):
            # eg: item: (6362, 
            #            '这只是...不活得还。', 
            #            '这只是...不活得还。', 
            #            [(6, '例'), (7, '子'), (9, '她'), (24, '心')]
            #            )
            data_sample = {}
            src_norm = item[1][:self.max_seq_len - 2]
            trg_norm = item[2][:self.max_seq_len - 2]
            # if config['normalize'] == True: # todo 直接截断
            #     src_norm = normalizer.normalize_str(item[1])[:self.max_seq_len-2]
            #     trg_norm = normalizer.normalize_str(item[2])[:self.max_seq_len-2]
            # else:
            #     src_norm = item[1][:self.max_seq_len - 2]
            #     trg_norm = item[2][:self.max_seq_len - 2]
            assert len(src_norm) == len(trg_norm)

            # src_token_list = list(src_norm)
            # trg_token_list = list(trg_norm)
            src_token_list = src_norm
            trg_token_list = trg_norm

            src_token_list.insert(0, '[CLS]')
            src_token_list.append('[SEP]')
            trg_token_list.insert(0, '[CLS]')
            trg_token_list.append('[SEP]')
            data_sample['id'] = item[0]
            data_sample['src_text'] = item[1]
            data_sample['input_ids'] = self.tokenizer.convert_tokens_to_ids(src_token_list)
            data_sample['token_type_ids'] = [0 for i in range(len(src_token_list))]
            data_sample['attention_mask'] = [1 for i in range(len(src_token_list))]
            data_sample['trg_ids'] = self.tokenizer.convert_tokens_to_ids(trg_token_list)
            data_sample['trg_text'] = item[2]
            data_sample['modification'] = item[3]

            data.append(data_sample)

        return data

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.LongTensor(self.data[idx]['input_ids'])
        item['token_type_ids'] = torch.LongTensor(self.data[idx]['token_type_ids'])
        item['attention_mask'] = torch.LongTensor(self.data[idx]['attention_mask'])
        if self.subset != 'test':
            item['trg_ids'] = torch.LongTensor(self.data[idx]['trg_ids'])
        return item

    def __len__(self):
        return len(self.data)

def csc_metrics(pred, gold):
    char_metrics = get_char_metrics(pred, gold)
    sent_metrics = get_sent_metrics(pred, gold)
    return char_metrics, sent_metrics

def get_dataloader_from_jsonl(path, config, subset, tokenizer):

    sub_dataset = CBZ_Dataset(path, config, subset, tokenizer)

    if subset == "train":
        is_shuffle = True
    else:
        is_shuffle = False

    collate_fn = Padding_in_batch(tokenizer.pad_token_id)

    data_loader = DataLoader(
        sub_dataset,
        batch_size=config['batch_size'],
        shuffle=is_shuffle,
        collate_fn=collate_fn
    )

    return data_loader

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.max_seq_len = self.config.get('max_seq_len', 128)
        self.seed = self.config.get('seed', 2021)
        self.num_epochs = self.config.get('num_epochs', 30)
        self.batch_size = self.config.get('batch_size', 48)
        self.freeze_bert = self.config.get('freeze_bert', False)
        self.tie_cls_weight = self.config.get('tie_cls_weight', True)
        self.label_ignore_id = self.config.get('label_ignore_id', 0)
        self.lr = self.config.get('lr', 2e-5)
        self.normalize = self.config.get('normalize', True)
        self.patience = self.config.get('patience', 10)
        self.save_path = self.config['save_path']
        self.best_score = {"valid-c": 0, "valid-s": 0}
        self.best_epoch = {"valid-c": 0, "valid-s": 0}
    
    def init(self):
        self.config['max_seq_len'] = self.max_seq_len
        self.config['seed'] = self.seed
        self.config['num_epochs'] = self.num_epochs
        self.config['batch_size'] = self.batch_size
        self.config['freeze_bert'] = self.freeze_bert
        self.config['tie_cls_weight'] = self.tie_cls_weight
        self.config['label_ignore_id'] = self.label_ignore_id
        self.config['lr'] = self.lr
        self.config['patience'] = self.patience
        self.config['normalize'] = self.normalize

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        jsondump(self.config, os.path.join(self.save_path, "train_config.json"))

        self.fix_seed(config['seed'])
        print(self.config)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config['pretrained_model'])
        self.model = BERT_Model(config, config['freeze_bert'], config['tie_cls_weight'])
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config['lr'])
        
        vocab_path = os.path.join(config['pretrained_model'], "vocab.txt")
        self.vocab = []
        with open(vocab_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            self.vocab.append(line.strip())
        print("加载训练、验证数据。。。")
        self.train_dataloader = get_dataloader_from_jsonl(config['train_path'], config, "train", self.tokenizer)
        self.valid_dataloader = get_dataloader_from_jsonl(config['dev_path'], config, "dev", self.tokenizer)
        self.scheduler = self.set_scheduler()

    def fix_seed(self, seed_num):
        torch.manual_seed(seed_num)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed(seed_num)

    def set_scheduler(self):
        num_epochs = self.config['num_epochs']
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0.1 * num_training_steps,
            num_training_steps=num_training_steps
        )
        return lr_scheduler

    def train(self):
        no_improve = 0
        # todo 循环 epoch
        for epoch in range(1, self.config['num_epochs'] + 1):
            # todo 训练
            self.model.train()
            train_loss, _ = self.__forward_prop(self.train_dataloader, back_prop=True)
            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_output = self.__forward_prop(self.valid_dataloader, back_prop=False)
            print(f"train_loss: {train_loss}, valid_loss: {valid_loss}")
            
            # todo 保存 训练完每个epoch 验证集识别结果
            if not os.path.exists(self.config['save_path'] + '/tmp/'):
                os.makedirs(self.config['save_path'] + '/tmp/')
            self.save_decode_result_para(valid_output, self.valid_dataloader.dataset.data,
                                    self.config['save_path'] + '/tmp/' + "valid_" + str(epoch) + ".txt")
            self.save_decode_result_lbl(valid_output, self.valid_dataloader.dataset.data,
                                   self.config['save_path'] + '/tmp/' + "valid_" + str(epoch) + ".lbl")
            
            # 计算指标
            results = self.get_result_para(valid_output, self.valid_dataloader.dataset.data)
            bzlist, prelist = self.get_bz_pr(results)
            # 句子指标
            juf1 = self.get_sen_prf(bzlist, prelist)
            # 字符指标
            charf1 = self.get_char_prf(results)

            if charf1['Correction'] > self.best_score['valid-c']:
                self.best_score['valid-c'] = charf1['Correction']
                self.best_epoch['valid-c'] = epoch
            
            if juf1['Correction'] > self.best_score['valid-s']:
                self.best_score['valid-s'] = juf1['Correction']
                self.best_epoch['valid-s'] = epoch

            # if max(self.best_epoch.values()) == epoch:
            #     self.__save_ckpt(epoch)
            self.__save_ckpt(epoch)
            
            print(f"curr epoch: {epoch} | curr best epoch {self.best_epoch}")
            print(f"best socre:{self.best_score}")
            print(f"no improve: {epoch - max(self.best_epoch.values())}")
            
            # if (epoch - max(self.best_epoch.values())) >= self.config['patience']:
            #     break

    def __forward_prop(self, dataloader, back_prop=True):
        # todo 正式 训练
        loss_sum = 0
        steps = 0
        collected_outputs = []
        # 循环 batch
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, logits = self.model(**batch)
            outputs = torch.argmax(logits, dim=-1)
            for outputs_i in outputs:
                collected_outputs.append(outputs_i)
            loss_sum += loss.item()
            if back_prop:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            steps += 1
        epoch_loss = loss_sum / steps
        return epoch_loss, collected_outputs

    def __save_ckpt(self, epoch):
        save_path = self.config['save_path']
        path = os.path.join(save_path, self.config['tag'] + f"-epoch-{epoch}.pt")
        torch.save(self.model.state_dict(), path)
    
    def save_decode_result_para(self, decode_pred, data, path):
        f = open(path, "w")
        results = []
        for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
            src_i = src['input_ids']
            line = ""
            pred_i = pred_i[:len(src_i)]
            pred_i = pred_i[1:-1]
            for id, ele in enumerate(pred_i):
                if self.vocab[ele] != "[UNK]":
                    line += self.vocab[ele]
                else:
                    line += src['src_text'][id]
            f.write("input:" + "".join(src['src_text']) + "\n")
            f.write("inference:" + line + "\n")
            f.write("trg:" + "".join(src['trg_text']) + "\n\n")
        f.close()
    
    def get_result_para(self, decode_pred, data):
        results = []
        for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
            src_i = src['input_ids']
            line = []
            pred_i = pred_i[:len(src_i)]
            pred_i = pred_i[1:-1]
            for id, ele in enumerate(pred_i):
                line.append(self.vocab[ele])
                # # if self.vocab[ele] != "[UNK]":
                # if len(self.vocab[ele]) == 1:
                #     line += self.vocab[ele]
                # else:
                #     line += src['src_text'][id]
            results.append((src['src_text'], src['trg_text'], line))
        return results
    
    def save_decode_result_lbl(self, decode_pred, data, path):
      with open(path, "w") as fout:
        for pred_i, src in zip(decode_pred, data):
            src_i = src['input_ids']
            line = src['id'] + ", "
            pred_i = pred_i[:len(src_i)]
            no_error = True
            for id, ele in enumerate(pred_i):
                if ele != src_i[id]:
                    if self.vocab[ele] != "[UNK]":
                        no_error = False
                        line += (str(id) + ", " + self.vocab[ele] + ", ")
            if no_error:
                line += '0'
            line = line.strip(", ")
            fout.write(line + "\n")
  
    def get_sen_prf(self, bzlist, prelist):
        ''' 计算句子指标 '''
        assert len(bzlist) == len(prelist)
        print("=" * 10 + " Sentence Level " + "=" * 10)
        # 检测指标
        tp, tp_c, targ_p, pred_p, hit, hit_c = 0, 0, 0, 0, 0, 0
        for pred_item, targ_item in zip(prelist, bzlist):
            pred, targ = sorted(pred_item), sorted(targ_item)
            if targ != []: # 标注有错字
                targ_p += 1
            if pred != []: # 识别有错字
                pred_p += 1
            if len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)): # 所有错误位置都相同，包含正确的句子
                hit += 1
            if pred == targ:
                hit_c += 1
            if pred != [] and len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)): # 识别有错字 所有错误位置都相同，
                tp += 1
            if pred != [] and pred == targ:
                tp_c += 1
        pre,cal,f1_d = calculate_pcf(targ_p, pred_p, tp)
        Acc = hit/len(bzlist)
        print(f"Detection:\nAccuracy: {Acc:.2f}, Precision: {pre:.2f}, Recall: {cal:.2f}, F1: {f1_d:.2f}")
        # 纠正指标
        pre,cal,f1_c = calculate_pcf(targ_p, pred_p, tp_c)
        Acc = hit_c/len(bzlist)
        print(f"Correction:\nAccuracy: {Acc:.2f}, Precision: {pre:.2f}, Recall: {cal:.2f}, F1: {f1_c:.2f}\n")
        return {"Detection": f1_d, "Correction":f1_c}

    def get_char_prf(self, results):
        bzlist, prelist = self.get_bz_pr(results)
        ''' 计算字符指标 '''
        assert len(bzlist) == len(prelist)
        TP = 0 # 检测对
        FP = 0 # 多检测
        FN = 0 # 未检测
        wrong_char = 0 # 标注错字个数
        print("=" * 10 + " Character Level " + "=" * 10)
        # 检测指标
        all_predict_true_index = [] # 全都检测对 
        all_gold_index = [] # 标注id 
        for pred_item, targ_item in zip(prelist, bzlist):
            gold_index = [x[0] for x in targ_item]
            all_gold_index.append(gold_index)
            predict_index = [x[0] for x in pred_item]
            each_true_index = []
            for i in predict_index:
                if i in gold_index:
                    TP += 1
                    each_true_index.append(i)
                else:
                    FP += 1
            for i in gold_index:
                wrong_char += 1 
                if i not in predict_index:
                    FN += 1
            all_predict_true_index.append(each_true_index)
                
        detection_precision, detection_recall, detection_f1 = calculate_pcf(TP+FN, TP+FP, TP) 
        print(f"Detection:\nPrecision: {detection_precision:.2f}, Recall: {detection_recall:.2f}, F1: {detection_f1:.2f}")

        # 纠正指标
        TP = 0 # 检测对
        FP = 0 # 多检测
        FN = 0 # 未检测
        for i,predict_true_index in enumerate(all_predict_true_index): # 只算位置识别正确的效果
            if len(predict_true_index) > 0: # 如果存在 正确 id
                predict_words = []
                for j in predict_true_index: # 循环句内错误 id
                    predict_words.append(results[i][2][j])
                    if results[i][1][j] == results[i][2][j]: # 标注id == 识别id
                        TP += 1
                    else:
                        FP += 1 # 虽然位置准确，但改错
            
                for j in all_gold_index[i]: 
                    # 用这个唯一区别就是：不管位置有没有识别对，但是 标注字 没在 识别里就算 FN
                    # 感觉有点问题，应该是连同 id 也计算进去，但是保持和举办方一致
                    if results[i][1][j] not in predict_words:
                        FN += 1
                    
        pre,cal,f1_c = calculate_pcf(TP+FN, TP+FP, TP)
        print(f"Correction:\nPrecision: {pre:.2f}, Recall: {cal:.2f}, F1: {f1_c:.2f}\n")
        return {"Detection": detection_f1, "Correction":f1_c}

    def get_bz_pr(self, results):
        all_predict_index = []
        all_gold_index = []
        for item in results:
            src, tgt, predict = item # 获取句子
            gold_index = []
            predict_index = []
            for i in range(len(predict)):
                if src[i] != tgt[i]: # 如果输入字符 不等于 标注字符 统计
                    gold_index.append((i, tgt[i])) # 标注错字 id
                if src[i] != predict[i]: # 如果输入字符 不等于 预测字符 统计
                    predict_index.append((i, predict[i])) # 识别错字 id
        
            all_gold_index.append(gold_index)
            all_predict_index.append(predict_index)
        return all_gold_index, all_predict_index
                

if __name__ == "__main__":

    # config = {
    #     'pretrained_model': "", # 预训练模型地址，采用的是 bert_base
    #     'train_path': "", # 训练集地址
    #     'dev_path': '', # 验证集地址
    #     'save_path': '', # 模型存放地址
    #     'tag': 'lang8', # 模型标识
    #     'num_epochs': 100,
    #     'batch_size': 96,
    #     'tie_cls_weight': True
    # }

    config = {
        'pretrained_model': "/home/hufei/pyapp/miaoxie_xiuci/auto_text_classifier/atc/data/bert_base_chinese",
        'train_path': "/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas_bju/train_data/lang8_oneanswer/train.jsonl",
        'dev_path': '/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas_bju/train_data/lang8_oneanswer/dev.jsonl',
        'save_path': '/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/代码提交内容/models/bju',
        'tag': 'lang8',
        'num_epochs': 100,
        'batch_size': 96,
        'tie_cls_weight': True
    }

    trainer = Trainer(config)
    trainer.init()
    trainer.train()
