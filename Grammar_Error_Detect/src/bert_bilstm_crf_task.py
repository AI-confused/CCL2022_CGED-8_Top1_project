import os
import random
import logging
import tqdm
import torch
import json
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from transformers import BertConfig, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, \
    AutoConfig, DebertaTokenizer
from easy_task.base_module import *
from model import BertBiLstmCrfForNer, CGEDModel, BERTChineseCharacterTokenizer, BertCrfForNer, FGM, PGD, BertCrfForNerRdrop
from Grammar_Error_Detect.src.bert_bilstm_crf_result import GrammarDetectResult



class GrammarDetectTask(BasePytorchTask):
    def __init__(self, task_setting: TaskSetting, load_train: bool=False, load_dev: bool=False, load_test: bool=False):
        """Custom Task definition class(custom).

        @task_setting: hyperparameters of Task.
        @load_train: load train set.
        @load_dev: load dev set.
        @load_test: load test set.
        """
        super(GrammarDetectTask, self).__init__(task_setting)
        self.logger.info('Initializing {}'.format(self.__class__.__name__))

        # BIEOS标签
        self.position_id2label = {index: value for index, value in enumerate(['O', 'S-B', 'W-B', 'R-B', 'S-I', 'W-I', 'R-I', 'S-E', 'W-E', 'R-E', 'S-S', 'M-S', 'R-S'])}
        self.position_label2id = {value: index for index, value in enumerate(['O', 'S-B', 'W-B', 'R-B', 'S-I', 'W-I', 'R-I', 'S-E', 'W-E', 'R-E', 'S-S', 'M-S', 'R-S'])}

        # self.identification_id2label = {index: value for index, value in enumerate(['O', 'W', 'S', 'R', 'M'])}
        # self.identification_label2id = {value: index for index, value in enumerate(['O', 'W', 'S', 'R', 'M'])}
        self.setting.num_label = len(self.position_label2id)

        # prepare model
        self.prepare_task_model()
        # self._decorate_model()

        # load dataset
        self.load_data(load_train, load_dev, load_test)

        # prepare optim
        if load_train:
            self.prepare_optimizer()

        self._decorate_model()

        # best score and output result(custom)
        self.best_dev_score = 0.0
        self.best_dev_epoch = 0
        self.output_result = {'result_type': '', 'task_config': self.setting.__dict__, 'result': []}


    def prepare_task_model(self):
        """Prepare ner task model(custom).

        Can be overwriten.
        """
        self.tokenizer = BERTChineseCharacterTokenizer.from_pretrained(self.setting.model_name)
        self.bert_config = BertConfig.from_pretrained(self.setting.model_name, num_labels=self.setting.num_label)
        self.setting.vocab_size = len(self.tokenizer.vocab)
        self.model = BertBiLstmCrfForNer(model_name=self.setting.model_name, config=self.bert_config, dropout=self.setting.dropout, rnn_layers=1)


    def prepare_optimizer(self):
        """Prepare ner task optimizer(custom).

        Can be overwriten.
        """
        # 冻结部分参数
        if hasattr(self.setting, 'freeze_layers'):
            for name, param in self.model.named_parameters():
                for layer_name in self.setting.freeze_layers:
                    if layer_name in name:
                        param.requires_grad = False
                        break
            # 定义一个fliter，只传入requires_grad=True的模型参数
            self.optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, self.model.parameters()), lr=float(self.setting.learning_rate))
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.setting.learning_rate))

        num_train_steps = int(len(self.train_features) / self.setting.train_batch_size / self.setting.gradient_accumulation_steps * self.setting.num_train_epochs)

        # scheduler
        if hasattr(self.setting, 'scheduler') and self.setting.scheduler=='linear':
            self.logger.info('================ do scheduler linear ================')
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(self.setting.warmup_portion*num_train_steps), num_training_steps=num_train_steps
            )
        elif hasattr(self.setting, 'scheduler') and self.setting.scheduler=='cosine':
            self.logger.info('================ do scheduler cosine ================')
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(self.setting.warmup_portion*num_train_steps), num_training_steps=num_train_steps, num_cycles=0.5
            )
        else:
            self.scheduler = None

        # adverse train
        if hasattr(self.setting, 'adverse_train') and self.setting.adverse_train=='fgm':
            self.logger.info('================ do adverse train fgm ================')
            self.adverse_attack = FGM(self.model)
        elif hasattr(self.setting, 'adverse_train') and self.setting.adverse_train=='pgd':
            self.logger.info('================ do adverse train pgd ================')
            self.adverse_attack = PGD(self.model)
        else:
            self.adverse_attack = None


    def prepare_result_class(self):
        """Prepare result calculate class(custom).

        Can be overwriten.
        """
        self.result = GrammarDetectResult(task_name=self.setting.task_name, id2label=self.position_id2label)


    def load_examples_features(self, data_type: str, file_name: str) -> tuple:
        """Load examples, features and dataset(custom).

        Can be overwriten, but with the same input parameters and output type.
        
        @data_type: train or dev or test
        @file_name: dataset file name
        """
        cached_features_file0 = os.path.join(self.setting.model_dir, 'cached_{}_{}_{}'.format(self.setting.percent, data_type, 'examples'))
        cached_features_file1 = os.path.join(self.setting.model_dir, 'cached_{}_{}_{}'.format(self.setting.percent, data_type, 'features'))

        if not self.setting.over_write_cache and os.path.exists(cached_features_file0) and os.path.exists(cached_features_file1):
            examples = torch.load(cached_features_file0)
            features = torch.load(cached_features_file1)
        else:
            examples = self.read_examples(os.path.join(self.setting.data_dir, file_name), percent=self.setting.percent)
            # torch.save(examples, cached_features_file0)
            features = self.convert_examples_to_features(examples,
                                                        tokenizer=self.tokenizer,
                                                        max_seq_len=self.setting.max_seq_len)
 
            # torch.save(features, cached_features_file1)
        dataset = TextDataset(features)
        return (examples, features, dataset, features[0].max_seq_len)


    def read_examples(self, input_file: str, percent: float=1.0) -> list:
        """Read data from a data file and generate list of InputExamples(custom).

        Can be overwriten, but with the same input parameters and output type.

        @input_file: data input file abs dir
        @percent: percent of reading samples
        """
        examples=[]

        data = open(input_file, 'r').readlines()

        if percent != 1.0:
            data = random.sample(data, int(len(data)*percent))

        for line in tqdm.tqdm(data, total=len(data), desc='read examples'):
            line = json.loads(line)

            if line['error'] == []:
                detect_label = 0
                identification_label = []
            else:
                detect_label = 1

                identification_label = []
                for item in line['error']:
                    identification_label.append(item['type'])
 
            examples.append(InputExample(
                                file_name=line['file_name'],
                                doc_id=line['id'],
                                text=line['text'],
                                correct=line['correct'],
                                detect_label=detect_label,
                                identification_label=identification_label,
                                position_label=line['error']
            ))            

        return examples


    def convert_examples_to_features(self, examples: list, tokenizer: BertTokenizer, max_seq_len: int, **kwargs) -> list:
        """Process the InputExamples into InputFeatures that can be fed into the model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples: list of InputExamples
        @tokenizer: class BertTokenizer or its inherited classes
        @max_seq_len: max length of tokenized text
        """
        features = []
        for example in tqdm.tqdm(examples, total=len(examples), desc='convert features'):
            if example.position_label != []:
                # tag label
                labels = sorted(example.position_label, key=lambda x: x['start'])
                sen_labels = []
                last_point = 0

                # 给序列打标签
                for label in labels:
                    if label['start'] < last_point:
                        continue
                    sen_labels += [0]*(label['start']-last_point)
                    try:
                        assert label['end'] - label['start'] >= 1
                    except:
                        # print('error1:', example.text, len(example.text), example.position_label, len(sen_labels))
                        break
                    if label['type'] == 'M':
                        sen_labels += [self.position_label2id[label['type']+'-S']]
                        last_point = label['start']+1
                    else:
                        if label['end'] - label['start'] == 1 and label['type'] != 'W':
                            # S
                            sen_labels += [self.position_label2id[label['type']+'-S']]
                        elif label['end'] - label['start'] == 1 and label['type'] == 'W':
                            print(label)
                        elif label['end'] - label['start'] == 2:
                            # BE
                            sen_labels += [self.position_label2id[label['type']+'-B'], self.position_label2id[label['type']+'-E']]
                        else:
                            # BIE
                            sen_labels += [self.position_label2id[label['type']+'-B']]
                            sen_labels += [self.position_label2id[label['type']+'-I']] * (label['end'] - label['start'] - 2)
                            sen_labels += [self.position_label2id[label['type']+'-E']]
                        last_point = label['end']

                sen_labels += [0] * (len(example.text) - last_point)
                try:
                    assert len(example.text) == len(sen_labels)
                except:
                    # print('error2:', example.text, len(example.text), example.position_label, len(sen_labels))
                    continue
            else:
                sen_labels = [0] * len(example.text)

            # tokenize
            tokenize_out = self.tokenizer(
                example.text,
                max_length=self.setting.max_seq_len,
                # return_offsets_mapping=self.setting.return_offsets_mapping
            )

            seq_len = len(sen_labels[:self.setting.max_seq_len-2])
            sen_labels = [-1] + sen_labels[:self.setting.max_seq_len-2] + [-1]

            assert len(sen_labels) == len(tokenize_out['input_ids'])

            features.append(
                InputFeature(
                    doc_id=example.doc_id,
                    file_name=example.file_name,
                    text=example.text[:self.setting.max_seq_len-2],
                    detect_label=example.detect_label,
                    identification_label=example.identification_label,
                    position_label=example.position_label,
                    input_ids=tokenize_out['input_ids'],
                    attention_mask=tokenize_out['attention_mask'],
                    token_type_ids=tokenize_out['token_type_ids'],
                    # offset_mapping=tokenize_out['offset_mapping'],
                    # sequence_ids=tokenize_out.sequence_ids(),
                    seq_len=seq_len, 
                    label=sen_labels[:],
                    max_seq_len=max_seq_len,
                )
            )

        return features


    def train(self, resume_base_epoch: int=None, resume_model_path: str=None):
        """Task level train func.

        @resume_base_epoch(int): start training epoch
        """
        self.logger.info('=' * 20 + 'Start Training {}'.format(self.setting.task_name) + '=' * 20)

        # resume model when restarting
        if resume_base_epoch is not None and resume_model_path is not None:
            raise ValueError('resume_base_epoch and resume_model_path can not be together!')
        elif resume_model_path is not None:
            self.logger.info('Training starts from other model: {}'.format(resume_model_path))
            self.resume_checkpoint(cpt_file_path=resume_model_path, resume_model=True, resume_optimizer=True)
            resume_base_epoch = 0
        else:
            if resume_base_epoch is None:
                if self.setting.resume_latest_cpt:
                    resume_base_epoch = self.get_latest_cpt_epoch()
                else:
                    resume_base_epoch = 0

            # resume cpt if possible
            if resume_base_epoch > 0:
                self.logger.info('Training starts from epoch {}'.format(resume_base_epoch))
                self.resume_checkpoint(cpt_file_name='{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                    self.setting.task_name, resume_base_epoch, resume_base_epoch, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed), resume_model=True, resume_optimizer=True)
            else:
                self.logger.info('Training starts from scratch')

        # prepare data loader
        self.train_dataloader = self._prepare_data_loader(self.train_dataset, self.setting.train_batch_size, rand_flag=True, collate_fn=self.custom_collate_fn_train)

        # do base train
        self._base_train(base_epoch_idx=resume_base_epoch)

        # save best score
        self.output_result['result'].append('best_dev_epoch: {} - best_dev_score: {}'.format(self.best_dev_epoch, self.best_dev_score))
        # self.output_result['result'].append('best_test_epoch: {} - best_test_score: {}'.format(self.best_test_epoch, self.best_test_score))

        # write output results
        self.write_results()

    
    def eval(self, global_step):
        """Task level eval func.

        @epoch(int): eval epoch
        """        
        data_type = 'dev'
        features = self.dev_features
        examples = self.dev_examples
        dataset = self.dev_dataset

        # prepare data loader
        self.eval_dataloader = self._prepare_data_loader(dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

        # init result calculate class
        self.prepare_result_class()

        # do base eval
        self._base_eval(global_step, data_type, examples, features)

        # calculate result score
        score = self.result.get_score()
        self.logger.info(json.dumps(score, indent=2, ensure_ascii=False))

        # return bad case in train-mode
        if self.setting.bad_case:
            self.return_selected_case(type_='dev_bad_case_pos', items=self.result.bad_case_pos, data_type='dev', epoch=global_step, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_iden', items=self.result.bad_case_iden, data_type='dev', epoch=global_step, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_det', items=self.result.bad_case_det, data_type='dev', epoch=global_step, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_FPR', items=self.result.bad_case_FPR, data_type='dev', epoch=global_step, file_type='excel')

        # return all result
        self.return_selected_case(type_='eval_prediction', items=self.result.all_result, data_type=data_type, epoch=global_step, file_type='excel')
        
        # save each epoch result
        self.output_result['result'].append('data_type: {} - checkpoint: {} - train_loss: {} - epoch_score: {}'\
                                            .format(data_type, global_step, self.train_loss, json.dumps(score, indent=2, ensure_ascii=False)))

        # save best model with specific standard(custom)
        if data_type == 'dev' and score['detect_f1']+score['iden_f1']+score['posi_f1']-score['FPR'] > self.best_dev_score:
            self.best_dev_step = global_step
            # self.best_dev_score = score[self.setting.evaluation_metric]
            self.best_dev_score = score['detect_f1']+score['iden_f1']+score['posi_f1']-score['FPR']
            self.logger.info('saving best dev model [{}]...'.format(self.best_dev_score))
            self.save_checkpoint(cpt_file_name='{}.cpt.{}.{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, data_type, 0, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))
            
        save_cpt_file = '{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, global_step, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed)
        if self.setting.save_cpt_flag == 1 and not os.path.exists(os.path.join(self.setting.model_dir, save_cpt_file)):
            # save last epoch
            last_checkpoint = self.get_latest_cpt_epoch()
            if last_checkpoint != 0:
                # delete lastest epoch model and store this epoch
                delete_cpt_file = '{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                    self.setting.task_name, last_checkpoint, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed)

                if os.path.exists(os.path.join(self.setting.model_dir, delete_cpt_file)):
                    os.remove(os.path.join(self.setting.model_dir, delete_cpt_file))
                    self.logger.info('remove model {}'.format(delete_cpt_file))
                else:
                    self.logger.info("{} does not exist".format(delete_cpt_file))

            self.logger.info('saving latest epoch model...')
            self.save_checkpoint(cpt_file_name='{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, global_step, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))

        elif self.setting.save_cpt_flag == 2 and not os.path.exists(os.path.join(self.setting.model_dir, save_cpt_file)):
            # save each epoch
            self.logger.info('saving checkpoint {} model...'.format(global_step))
            self.save_checkpoint(cpt_file_name='{}.cpt.checkpoint{}.e({}).b({}).p({}).s({})'.format(\
                self.setting.task_name, global_step, self.setting.num_train_epochs, self.setting.train_batch_size, str(self.setting.percent).replace('.','。'), self.setting.seed))


    def custom_collate_fn_train(self, features: list) -> list:
        """Convert batch training examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        input_ids = pad_sequence([torch.tensor(feature.input_ids, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([torch.tensor(feature.attention_mask, dtype=torch.long) for feature in features], batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(feature.token_type_ids, dtype=torch.long) for feature in features], batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(feature.label, dtype=torch.long) for feature in features], batch_first=True, padding_value=-1)

        return [input_ids, attention_mask, token_type_ids, labels, features]


    def custom_collate_fn_eval(self, features: list) -> list:
        """Convert batch eval examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        input_ids = pad_sequence([torch.tensor(feature.input_ids, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([torch.tensor(feature.attention_mask, dtype=torch.long) for feature in features], batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(feature.token_type_ids, dtype=torch.long) for feature in features], batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(feature.label, dtype=torch.long) for feature in features], batch_first=True, padding_value=-1)

        return [input_ids, attention_mask, token_type_ids, labels, features]


    def resume_test_at(self, resume_model_path: str, **kwargs):
        """Resume checkpoint and do test(custom).

        Can be overwriten, but with the same input parameters.
        
        @resume_model_path: do test model path
        """
        # extract kwargs
        header = kwargs.pop("header", None)

        self.resume_checkpoint(cpt_file_path=resume_model_path, resume_model=True, resume_optimizer=False)

        # prepare data loader
        self.eval_dataloader = self._prepare_data_loader(self.test_dataset, self.setting.eval_batch_size, rand_flag=False, collate_fn=self.custom_collate_fn_eval)

        # init result calculate class
        self.prepare_result_class()

        # do test
        self._base_eval(0, 'test', self.test_examples, self.test_features)

        score = self.result.get_score()
        self.logger.info(json.dumps(score, indent=2, ensure_ascii=False))

        # return bad case in train-mode
        if self.setting.bad_case:
            self.return_selected_case(type_='test_bad_case_pos', items=self.result.bad_case_pos, data_type='test', epoch=0, file_type='excel')
            self.return_selected_case(type_='test_bad_case_iden', items=self.result.bad_case_iden, data_type='test', epoch=0, file_type='excel')
            self.return_selected_case(type_='test_bad_case_det', items=self.result.bad_case_det, data_type='test', epoch=0, file_type='excel')
            self.return_selected_case(type_='test_bad_case_FPR', items=self.result.bad_case_FPR, data_type='test', epoch=0, file_type='excel')

        # output test prediction
        self.return_selected_case(type_='test_prediction', items=self.result.all_result, data_type='test', epoch=0, file_type='excel')
    

    def get_result_on_batch(self, batch: tuple):
        """Return batch output logits during eval model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @batch: /
        """
        input_ids, attention_mask, token_type_ids, labels, features = batch
        logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).detach().cpu()
        return logits, labels.detach().cpu(), features


    def get_loss_on_batch(self, batch: tuple):
        """Return batch loss during training model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @batch: /
        """
        input_ids, attention_mask, token_type_ids, labels, features = batch
        loss = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        return loss
