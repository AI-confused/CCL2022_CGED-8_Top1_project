import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
from easy_task.base_module import *


class GrammarDetectResult(BaseResult):
    """Store and calculate result class(custom), inherit from BaseResult.

    @task_name: string of task name.
    @id2label: id to label map dict.
    @max_seq_len: max sequence length of input feature.
    """
    def __init__(self, task_name: str, id2label: dict):
        super(GrammarDetectResult, self).__init__(task_name=task_name)
        self.bad_case_pos = {'text': [], 'id': [], \
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.bad_case_iden = {'text': [], 'id': [], \
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.bad_case_det = {'text': [], 'id': [], \
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.bad_case_FPR = {'text': [], 'id': [], \
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.all_result = {'text': [], 'id': [], \
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.prediction = {}
        self.id2label = id2label


    def get_entity_span(self, label: int, start: int, logit: int, entity_type: str, text: str):
        """Get entity span end index.
        17分类解码
        @label:
        @start:
        @logit:
        @entity_type: S or BIE
        return
        @has entity: bool
        @entity_end: 实体的右边界
        @search_end: 解码范围的右边界
        """
        index = start+1
        if entity_type == 'BIE':
            span = []
            while index<len(logit):
                if logit[index] == label+4 or logit[index] == label+8:
                    # 把同一个标签的I和E先添加进去
                    span.append(logit[index])
                    index += 1
                else:
                    # 当遇到其他类型的标签时，开始分析已有的解码序列
                    if label+8 not in span: # 不存在E, 则返回-1
                        return False, -1, index
                    first_E_index = span.index(label+8)
                    return True, start+1+first_E_index+1, index
        elif entity_type == 'S':
            if start == len(logit)-1 or logit[start] != logit[index]:
                return True, index, index
            else:
                return False, -1, index
            # return True, index, index
        else:
            raise Exception('wrong entity_type')
        # 持续解码到了末尾
        if label+8 not in span: # 不存在E, 则返回-1
            return False, -1, index
        first_E_index = span.index(label+8)
        return True, start+1+first_E_index+1, index


    def decode_entity(self, logit: torch.Tensor, text: str):
        """Decode entitys from output logits and input texts.
          17分类的解码
        @logit: sequence tagging output of an input.
        @text: input sequence.
        """
        logit = logit.numpy().tolist()
        assert len(logit)==len(text)
        ans = []
        i = 0
        while i<len(logit):
            if logit[i] in range(1, 5): # 遇到B开头的实体
                has_entity, entity_right_end, search_end = self.get_entity_span(logit[i], i, logit, 'BIE', text)
                if has_entity:
                    key = self.id2label[logit[i]][0]
                    value = text[i:entity_right_end]
                    ans.append({'error_type': key, 'error_info': value, 'position': (i, entity_right_end)})
                i = search_end
            elif logit[i] in range(13, 17): # 遇到S的实体
                has_entity, entity_right_end, search_end = self.get_entity_span(logit[i], i, logit, 'S', text)
                if has_entity:
                    key = self.id2label[logit[i]][0]
                    value = text[i:entity_right_end]
                    ans.append({'error_type': key, 'error_info': value, 'position': (i, entity_right_end)})
                i = search_end
            else:
                i += 1 # 遇到O、I开头的实体、E开头的实体
        return ans
        
        
    def update_batch(self, batch_results: list, **kwargs):
        """Update batch data in custom task.

        @batch_results: [batch_logits, batch_labels, batch_features]
        """
        BIO_index, batch_labels, batch_features = batch_results
        for i in range(len(batch_features)):
            example = batch_features[i]
            # BIO_index = torch.max(batch_logits[i], dim=1)[1]
            # assert len(BIO_index)==self.max_seq_len
            self.all_result['id'].append(example.doc_id)
            self.all_result['text'].append(example.text)

            entities = self.decode_entity(BIO_index[i][1:example.seq_len+1], example.text[:example.seq_len])

            if example.doc_id not in self.prediction.keys():
                self.prediction[example.doc_id] = {}
            
            # detection pred
            if entities:
                self.prediction[example.doc_id]['detection'] = 1
            else:
                self.prediction[example.doc_id]['detection'] = 0
            self.all_result['detection_pred'].append(self.prediction[example.doc_id]['detection'])

            # detection label
            label = batch_labels[i][1:example.seq_len+1].detach().tolist()
            # if label == [0]*len(label):
            if example.detect_label == 0:
                assert label == [0]*len(label)
                self.prediction[example.doc_id]['detection_label'] = 0
            else:
                self.prediction[example.doc_id]['detection_label'] = 1
            self.all_result['detection_label'].append(self.prediction[example.doc_id]['detection_label'])

            # identification pred
            if entities:
                self.prediction[example.doc_id]['identification'] = list(set([entity['error_type'] for entity in entities]))
            else:
                self.prediction[example.doc_id]['identification'] = []
            self.all_result['identification_pred'].append(self.prediction[example.doc_id]['identification'])
            
            # identification label
            self.prediction[example.doc_id]['identification_label'] = list(set(example.identification_label))
            self.all_result['identification_label'].append(self.prediction[example.doc_id]['identification_label'])

            # position
            self.prediction[example.doc_id]['position'] = entities[:]
            self.all_result['position_pred'].append(entities[:])

            # position label
            self.prediction[example.doc_id]['position_label'] = [{'error_type': item['type'], \
                'error_info': example.text[:example.seq_len][item['start']: item['end']], \
                    'position': (item['start'], item['end'])} for item in example.position_label]
  
            self.all_result['position_label'].append(self.prediction[example.doc_id]['position_label'][:])

            # badcase
            # pos
            if self.prediction[example.doc_id]['position'] != self.prediction[example.doc_id]['position_label']:
                self.bad_case_pos['id'].append(example.doc_id)
                self.bad_case_pos['text'].append(example.text)
                # detection level
                self.bad_case_pos['detection_pred'].append(self.prediction[example.doc_id]['detection'])
                self.bad_case_pos['detection_label'].append(self.prediction[example.doc_id]['detection_label'])
                # identification level
                self.bad_case_pos['identification_pred'].append(self.prediction[example.doc_id]['identification'])
                self.bad_case_pos['identification_label'].append(self.prediction[example.doc_id]['identification_label'])
                # position level
                self.bad_case_pos['position_pred'].append(self.prediction[example.doc_id]['position'])
                self.bad_case_pos['position_label'].append(self.prediction[example.doc_id]['position_label'])
            # iden
            if self.prediction[example.doc_id]['identification'] != self.prediction[example.doc_id]['identification_label']:
                self.bad_case_iden['id'].append(example.doc_id)
                self.bad_case_iden['text'].append(example.text)
                # detection level
                self.bad_case_iden['detection_pred'].append(self.prediction[example.doc_id]['detection'])
                self.bad_case_iden['detection_label'].append(self.prediction[example.doc_id]['detection_label'])
                # identification level
                self.bad_case_iden['identification_pred'].append(self.prediction[example.doc_id]['identification'])
                self.bad_case_iden['identification_label'].append(self.prediction[example.doc_id]['identification_label'])
                # position level
                self.bad_case_iden['position_pred'].append(self.prediction[example.doc_id]['position'])
                self.bad_case_iden['position_label'].append(self.prediction[example.doc_id]['position_label'])
            # det
            if self.prediction[example.doc_id]['detection'] != self.prediction[example.doc_id]['detection_label']:
                self.bad_case_det['id'].append(example.doc_id)
                self.bad_case_det['text'].append(example.text)
                # detection level
                self.bad_case_det['detection_pred'].append(self.prediction[example.doc_id]['detection'])
                self.bad_case_det['detection_label'].append(self.prediction[example.doc_id]['detection_label'])
                # identification level
                self.bad_case_det['identification_pred'].append(self.prediction[example.doc_id]['identification'])
                self.bad_case_det['identification_label'].append(self.prediction[example.doc_id]['identification_label'])
                # position level
                self.bad_case_det['position_pred'].append(self.prediction[example.doc_id]['position'])
                self.bad_case_det['position_label'].append(self.prediction[example.doc_id]['position_label'])
            # FPR
            if self.prediction[example.doc_id]['detection']==1 and self.prediction[example.doc_id]['detection_label']==0:
                self.bad_case_FPR['id'].append(example.doc_id)
                self.bad_case_FPR['text'].append(example.text)
                # detection level
                self.bad_case_FPR['detection_pred'].append(self.prediction[example.doc_id]['detection'])
                self.bad_case_FPR['detection_label'].append(self.prediction[example.doc_id]['detection_label'])
                # identification level
                self.bad_case_FPR['identification_pred'].append(self.prediction[example.doc_id]['identification'])
                self.bad_case_FPR['identification_label'].append(self.prediction[example.doc_id]['identification_label'])
                # position level
                self.bad_case_FPR['position_pred'].append(self.prediction[example.doc_id]['position'])
                self.bad_case_FPR['position_label'].append(self.prediction[example.doc_id]['position_label'])
            
            # feature
            self.prediction[example.doc_id]['feature'] = example


    def get_score(self):
        """Calculate NER micro f1 score.
        """
        # calculate FPR
        FP= 0
        for i in range(len(self.all_result['detection_label'])):
            if self.all_result['detection_label'][i] == 0 and self.all_result['detection_pred'][i] == 1:
                FP += 1
        FPR = FP/(len([item for item in self.all_result['detection_label'] if item==0]) + 1e-20)

        # calculate detection metrics
        # detect_acc = accuracy_score(self.all_result['detection_label'], self.all_result['detection_pred'])
        detect_pre = precision_score(self.all_result['detection_label'], self.all_result['detection_pred'])
        detect_rec = recall_score(self.all_result['detection_label'], self.all_result['detection_pred'])
        detect_f1 = f1_score(self.all_result['detection_label'], self.all_result['detection_pred'])

        # calculate identification metrics
        TP, TN = 0, 0
        label_num = sum(len(item) for item in self.all_result['identification_label'])
        pred_num = sum(len(item) for item in self.all_result['identification_pred'])
        for i in range(len(self.all_result['identification_pred'])):
            if self.all_result['identification_pred'][i] == self.all_result['identification_label'][i] == []:
                TN += 1
            for pred in self.all_result['identification_pred'][i]:
                if pred in self.all_result['identification_label'][i]:
                    TP += 1
        # iden_acc = (TP+TN)/(pred_num+TN)
        iden_precision = TP/(pred_num+1e-20)
        iden_recall = TP/(label_num+1e-20)
        iden_f1 = 2*iden_precision*iden_recall/(iden_precision+iden_recall+1e-20)

        # calculate position metrics
        TP, TN = 0, 0
        label_num = sum([len(item) for item in self.all_result['position_label']])
        pred_num = sum([len(item) for item in self.all_result['position_pred']])
        for i in range(len(self.all_result['position_pred'])):
            if self.all_result['position_pred'][i] == self.all_result['position_label'][i] == []:
                TN += 1
            for pred in self.all_result['position_pred'][i]:
                if pred in self.all_result['position_label'][i]:
                    TP += 1
        # posi_acc = (TP+TN)/(pred_num+TN)
        posi_precision = TP/(pred_num+1e-20)
        posi_recall = TP/(label_num+1e-20)
        posi_f1 = 2*posi_precision*posi_recall/(posi_precision+posi_recall+1e-20)
        
        return {
            'FPR': FPR,
            # 'detect_acc': round(detect_acc, 4),
            'detect_pre': round(detect_pre, 4),
            'detect_rec': round(detect_rec, 4),
            'detect_f1': round(detect_f1, 4),
            # 'iden_acc': round(iden_acc, 4),
            'iden_precision': round(iden_precision, 4),
            'iden_recall': round(iden_recall, 4),
            'iden_f1': round(iden_f1, 4),
            # 'posi_acc': round(posi_acc, 4),
            'posi_precision': round(posi_precision, 4),
            'posi_recall': round(posi_recall, 4),
            'posi_f1': round(posi_f1, 4)
        }