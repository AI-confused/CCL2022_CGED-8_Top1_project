import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
from easy_task.base_module import *
import Levenshtein


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
            result.append((str(edit[1]+1), str(edit[1]+1), "M", tgt_line[edit[3]:edit[4]]))
        elif edit[0] == "replace":
            # new_op = post_process_S(src_line, (str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
            result.append((str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
        elif edit[0] == "delete":
            result.append((str(edit[1]+1), str(edit[2]), "R"))
        elif edit[0] == "hybrid":
            # new_op = post_process_S(src_line, (str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
            result.append((str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
        elif edit[0] == "luanxu":
            result.append((str(edit[1]+1), str(edit[2]) , "W"))

    return result


class GrammarCorrectResult(BaseResult):
    """Store and calculate result class(custom), inherit from BaseResult.

    @task_name: string of task name.
    @id2label: id to label map dict.
    @max_seq_len: max sequence length of input feature.
    """
    def __init__(self, task_name: str):
        super(GrammarCorrectResult, self).__init__(task_name=task_name)
        self.bad_case_pos = {'text': [], 'id': [], 'correct': [],\
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.bad_case_iden = {'text': [], 'id': [], 'correct': [],\
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.bad_case_det = {'text': [], 'id': [], 'correct': [],\
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.bad_case_FPR = {'text': [], 'id': [], 'correct': [],\
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.all_result = {'text': [], 'id': [], 'correct': [], 'edits': [],\
            'detection_pred': [], 'identification_pred': [], 'position_pred': [], \
                'detection_label': [], 'identification_label': [], 'position_label': []}
        self.prediction = {}

    # def despose_UNK(self, src, trg):
    #     try:
    #         while '[UNK]' in trg:
    #             if '[UNK][UNK][UNK]' in trg:
    #                 pos = trg.index('[UNK][UNK][UNK]')
    #                 if trg[pos+15: pos+17] in src:
    #                     nos = src.index(trg[pos+15: pos+17])
    #                     trg = trg[:pos] + src[nos-3: nos] + trg[pos+15:]
    #                 else:
    #                     print(src, trg)
    #                     # raise Exception('error')
    #             elif '[UNK][UNK]' in trg:
    #                 pos = trg.index('[UNK][UNK]')
    #                 if trg[pos+10: pos+12] in src:
    #                     nos = src.index(trg[pos+10: pos+12])
    #                     trg = trg[:pos] + src[nos-2: nos] + trg[pos+10:]
    #                 else:
    #                     print(src, trg)
    #             elif '[UNK]' in trg:
    #                 pos = trg.index('[UNK]')
    #                 if trg[pos+5: pos+7] in src:
    #                     nos = src.index(trg[pos+5: pos+7])
    #                     trg = trg[:pos] + src[nos-1] + trg[pos+5:]
    #                 else:
    #                     print(src, trg)
    #                     # raise Exception('error')
    #     except:
    #         pass
    #     return src, trg

        
    def update_batch(self, batch_results: list, **kwargs):
        """Update batch data in custom task.

        @batch_results: [batch_logits, batch_labels, batch_features]
        """
        batch_predicts, batch_features = batch_results
        for i in range(len(batch_features)):
            example = batch_features[i]
            # BIO_index = torch.max(batch_logits[i], dim=1)[1]
            # assert len(BIO_index)==self.max_seq_len
            self.all_result['id'].append(example.doc_id)
            self.all_result['text'].append(example.text)
            if batch_predicts[i] == ' ' or batch_predicts[i] == '':
                batch_predicts[i] = example.text
            # src, trg = self.despose_UNK(example.text, ''.join(batch_predicts[i].split()).replace('[BLK]', ''))
            
            self.all_result['correct'].append(''.join(batch_predicts[i].split()).replace('[BLK]', ''))

            entities = []
            edits = pair2edits_new(example.text+'。', ''.join(batch_predicts[i].split()).replace('[BLK]', '')+'。')
            self.all_result['edits'].append(edits)

            for edit in edits:
                assert len(edit) in [3, 4] and int(edit[1]) >= int(edit[0])
                # if edit[1] < edit[0]:
                #     print(edit)
                entities.append({'start': int(edit[0])-1, 'end': int(edit[1]), 'type': edit[2]})

            if example.doc_id not in self.prediction.keys():
                self.prediction[example.doc_id] = {}
            
            # detection pred
            if entities:
                self.prediction[example.doc_id]['detection'] = 1
            else:
                self.prediction[example.doc_id]['detection'] = 0
            self.all_result['detection_pred'].append(self.prediction[example.doc_id]['detection'])

            # detection label
            # label = batch_labels[i][1:example.seq_len+1].detach().tolist()
            if example.detect_label == 0:
                self.prediction[example.doc_id]['detection_label'] = 0
            else:
                self.prediction[example.doc_id]['detection_label'] = 1
            self.all_result['detection_label'].append(self.prediction[example.doc_id]['detection_label'])

            # identification pred
            if entities:
                self.prediction[example.doc_id]['identification'] = list(set([entity['type'] for entity in entities]))
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
            # self.prediction[example.doc_id]['position_label'] = self.decode_entity(label, example.text[:example.seq_len])
            self.prediction[example.doc_id]['position_label'] = example.position_label[:]
            # for index, p_l in enumerate(sorted(example.position_label, key=lambda x: x['start'])):
            #     try:
            #         assert p_l['start'] == self.prediction[example.doc_id]['position_label'][index]['position'][0]
            #         assert p_l['end'] == self.prediction[example.doc_id]['position_label'][index]['position'][1]
            #         assert p_l['type'] == self.prediction[example.doc_id]['position_label'][index]['error_type']
            #         assert example.text[:example.seq_len][p_l['start']:p_l['end']] == self.prediction[example.doc_id]['position_label'][index]['error_info']
            #     except:
            #         print(example.file_name, example.doc_id)
            self.all_result['position_label'].append(self.prediction[example.doc_id]['position_label'][:])

            # badcase
            # pos
            if self.prediction[example.doc_id]['position'] != self.prediction[example.doc_id]['position_label']:
                self.bad_case_pos['id'].append(example.doc_id)
                self.bad_case_pos['text'].append(example.text)
                self.bad_case_pos['correct'].append(''.join(batch_predicts[i].split()).replace('[BLK]', ''))
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
                self.bad_case_iden['correct'].append(''.join(batch_predicts[i].split()).replace('[BLK]', ''))
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
                self.bad_case_det['correct'].append(''.join(batch_predicts[i].split()).replace('[BLK]', ''))
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
                self.bad_case_FPR['correct'].append(''.join(batch_predicts[i].split()).replace('[BLK]', ''))
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


    # def get_prediction(self):
    #     """Obtain predictions in test-mode.
    #     """
    #     for _, value in self.prediction.items():
    #         res_pred = []
    #         if value['entity']:
    #             res_pred = self.reform_entity(value['entity'])

    #             # return prediction
    #             self.all_result['text'].append(value['feature'].sentence)
    #             self.all_result['id'].append(value['feature'].doc_id)
    #             self.all_result['pred'].append(res_pred[:])

    #     self.all_result.pop('label')