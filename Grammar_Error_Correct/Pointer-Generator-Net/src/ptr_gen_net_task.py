import os
import sys
sys.path.append('/home/liyunliang/CGED_Task/pointer-generator-pytorch/')
import random
import logging
import tqdm
import torch
import json
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import math
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizerFast, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, BartForConditionalGeneration, BertTokenizer, BertTokenizerFast
from easy_task.base_module import *
# from model import CGEDModel, BERTChineseCharacterTokenizer, BertCrfForNer, S2AModel
from model import PGNModel
from correct_result import GrammarCorrectResult
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_



class GrammarCorrectTask(BasePytorchTask):
    def __init__(self, task_setting: TaskSetting, load_train: bool=False, load_dev: bool=False, load_test: bool=False):
        """Custom Task definition class(custom).

        @task_setting: hyperparameters of Task.
        @load_train: load train set.
        @load_dev: load dev set.
        @load_test: load test set.
        """
        super(GrammarCorrectTask, self).__init__(task_setting)
        self.logger.info('Initializing {}'.format(self.__class__.__name__))

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
        self.cls_id, self.sep_id, self.unk_id = 101, 102, 100
        self.tokenizer = BertTokenizerFast.from_pretrained(self.setting.model_name)
        self.model_config = {
            'is_coverage': True,
            'vocab_size': len(self.tokenizer.vocab),
            'dropout': self.setting.dropout,
            'model_path': self.setting.model_name,
            # 'hidden_dim': 768,
            'cov_loss_wt': 1.0,
        }
        if 'base' in self.setting.model_name:
            self.model_config['hidden_dim'] = 768
        else:
            self.model_config['hidden_dim'] = 1024
        self.model = PGNModel(self.model_config)


    def prepare_optimizer(self):
        """Prepare ner task optimizer(custom).

        Can be overwriten.
        """
        # s2a_embed = list(map(id, self.model.s2a_embed.parameters()))
        # encoder_embed = list(map(id, self.model.encoder.embed.embed.parameters()))
        # decoder_embed = list(map(id, self.model.decoder.embed.embed.parameters()))
        # base_params = filter(lambda p: id(p) not in s2a_embed + encoder_embed + decoder_embed, self.model.parameters())
        # optimizer_parameters = [
        #     {'params': self.model.s2a_embed.parameters(), 'lr': 1e-5},
        #     {'params': self.model.encoder.embed.embed.parameters(), 'lr': 1e-5},
        #     {'params': self.model.decoder.embed.embed.parameters(), 'lr': 1e-5},
        #     {'params': base_params}]
            # {'params': self.model.fc1.parameters(), 'lr': float(self.setting.learning_rate)},
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.setting.learning_rate))
        # self.optimizer = torch.optim.Adam(optimizer_parameters,
        #                                     lr=float(self.setting.learning_rate),
        #                                     betas=(0.9, 0.998))

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
            # self.adverse_attack = FGM(self.model)
        elif hasattr(self.setting, 'adverse_train') and self.setting.adverse_train=='pgd':
            self.logger.info('================ do adverse train pgd ================')
            # self.adverse_attack = PGD(self.model)
        else:
            self.adverse_attack = None


    def prepare_result_class(self):
        """Prepare result calculate class(custom).

        Can be overwriten.
        """
        self.result = GrammarCorrectResult(task_name=self.setting.task_name)


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
            examples = self.read_examples(os.path.join(self.setting.data_dir, file_name), data_type=data_type, percent=self.setting.percent)
            # torch.save(examples, cached_features_file0)
            features = self.convert_examples_to_features(examples,
                                                        tokenizer=self.tokenizer,
                                                        max_seq_len=self.setting.max_seq_len,
                                                        data_type=data_type)
 
            # torch.save(features, cached_features_file1)
        dataset = TextDataset(features)
        return (examples, features, dataset, features[0].max_seq_len)


    def read_examples(self, input_file: str, data_type: str, percent: float=1.0) -> list:
        """Read data from a data file and generate list of InputExamples(custom).

        Can be overwriten, but with the same input parameters and output type.

        @input_file: data input file abs dir
        @percent: percent of reading samples
        """
        examples=[]

        data = open(input_file, 'r').readlines()

        # if percent != 1.0 and data_type == 'train':
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
        if kwargs['data_type'] == 'train':
            for example in tqdm.tqdm(examples, total=len(examples), desc='convert features'):
                encoder_input = self.tokenizer.encode(
                    example.text,
                    max_length=self.setting.max_seq_len,
                    add_special_tokens=True
                )

                decoder_output = self.tokenizer.encode(
                    example.correct,
                    max_length=self.setting.max_seq_len-1,
                    add_special_tokens=False
                ) + [self.sep_id]

                decoder_input = [self.cls_id] + decoder_output[:-1]

                assert len(decoder_output) == len(decoder_input)

                features.append(
                    InputFeature(
                        doc_id=example.doc_id,
                        ori=example.text,
                        cor=example.correct,
                        encoder_input=encoder_input,
                        decoder_input=decoder_input,
                        decoder_output=decoder_output,
                        max_seq_len=max_seq_len,
                    )
                )
        elif kwargs['data_type'] in ['dev', 'test']: 
            for example in tqdm.tqdm(examples, total=len(examples), desc='convert features'):
                encoder_input = self.tokenizer(
                    example.text,
                    max_length=self.setting.max_seq_len,
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                )

                features.append(
                    InputFeature(
                        doc_id=example.doc_id,
                        file_name=example.file_name,
                        text=example.text,
                        detect_label=example.detect_label,
                        identification_label=example.identification_label,
                        position_label=example.position_label,
                        encoder_input=encoder_input['input_ids'],
                        max_seq_len=max_seq_len,
                        offsets_mapping=encoder_input['offset_mapping']
                    )
                )
        else:
            raise Exception('Error')
        
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

    
    # def _base_train(self, **kwargs):
    #     """Base task train func with a set of parameters.

    #     This class should not be rewritten.
        
    #     @kwargs: base_epoch_idx
    #     """
    #     assert self.model is not None
    #     if self.num_train_steps is None:
    #         self.num_train_steps = int(len(self.train_features) / self.setting.train_batch_size / self.setting.gradient_accumulation_steps * self.setting.num_train_epochs)

    #     self.logger.info('='*20 + 'Start Base Training' + '='*20)
    #     self.logger.info("\tTotal examples Num = {}".format(len(self.train_examples)))
    #     self.logger.info("\tTotal features Num = {}".format(len(self.train_features)))
    #     self.logger.info("\tBatch size = {}".format(self.setting.train_batch_size))
    #     self.logger.info("\tNum steps = {}".format(self.num_train_steps))     

    #     # start training
    #     global_step = 0
    #     self.train_loss = 0

    #     for epoch_idx in tqdm.trange(kwargs['base_epoch_idx'], int(self.setting.num_train_epochs), desc="Epoch"):
    #         # self.model.train()
    #         tr_loss = 0
    #         nb_tr_examples, nb_tr_steps = 0, 0

    #         bar = tqdm.tqdm(self.train_dataloader)
    #         for step, batch in enumerate(bar):
    #             batch = self.set_batch_to_device(batch)
    #             loss = self.get_loss_on_batch(batch)

    #             if self.n_gpu > 1:
    #                 # mean() to average on multi-gpu.
    #                 loss = loss.mean()  
    #             if self.setting.gradient_accumulation_steps > 1:
    #                 loss = loss / self.setting.gradient_accumulation_steps

    #             # backward
    #             loss.backward()

    #             # 梯度裁剪
    #             # clip_grad_norm_(self.model.parameters(), 2.0)

    #             loss_scalar = loss.item()
    #             tr_loss += loss_scalar
    #             self.train_loss = round(tr_loss * self.setting.gradient_accumulation_steps / (nb_tr_steps+1), 4)
    #             bar.set_description('loss {}'.format(self.train_loss))
    #             nb_tr_examples += self.setting.train_batch_size
    #             nb_tr_steps += 1

    #             # 对抗训练
    #             if self.adverse_attack != None:
    #                 self.adverse_attack.attack() # 在embedding上添加对抗扰动
    #                 loss_adv = self.get_loss_on_batch(batch)
    #                 if self.n_gpu > 1:
    #                     loss_adv = loss_adv.mean()
    #                 if self.setting.gradient_accumulation_steps > 1:
    #                     loss_adv = loss_adv / self.setting.gradient_accumulation_steps
    #                 # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    #                 loss_adv.backward() 
    #                 # 恢复embedding参数
    #                 self.adverse_attack.restore() 

    #             if (step + 1) % self.setting.gradient_accumulation_steps == 0:
    #                 self.optimizer.step()
    #                 if hasattr(self.setting, 'scheduler') and self.setting.scheduler != None:
    #                     self.scheduler.step()
    #                 self.model.zero_grad()
    #                 global_step += 1

    #                 if global_step>1 and global_step % int(len(self.train_dataloader)//self.setting.gradient_accumulation_steps*self.setting.eval_portion) == 0:
    #                     # do epoch eval
    #                     self.eval(global_step)

    #                     self.model.train()

    
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


    def nopeak_mask(self, size, batch=1):
        """上三角mask矩阵
        """
        np_mask = np.triu(np.ones((batch, size, size)), k=1).astype('uint8')
        np_mask =  Variable(torch.from_numpy(np_mask) == 0)
        return np_mask


    def create_masks(self, src, trg):
        # encoder对pad部分进行mask
        src_mask = (src != self.tokenizer.pad_token_id)

        if trg is not None:
            # decoder对pad进行mask
            trg_mask = (trg != self.tokenizer.pad_token_id)
            # get seq_len for matrix
            # size = trg.size(1)
            # # assert size == self.setting.max_seq_len
            # np_mask = self.nopeak_mask(size)
            # # 结合padding部分和peakmask部分
            # trg_mask = trg_mask & np_mask
        else:
            trg_mask = None
        return src_mask, trg_mask


    def custom_collate_fn_train(self, features: list) -> list:
        """Convert batch training examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        encoder_input = pad_sequence([torch.tensor(feature.encoder_input, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # s2a_input = pad_sequence([torch.tensor(feature.s2a_input, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_input = pad_sequence([torch.tensor(feature.decoder_input, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_output = pad_sequence([torch.tensor(feature.decoder_output, dtype=torch.long) for feature in features], batch_first=True, padding_value=-1)
        ori_mask, cor_mask = self.create_masks(encoder_input, decoder_input)

        return [encoder_input, decoder_input, decoder_output, ori_mask, cor_mask, features]


    def custom_collate_fn_eval(self, features: list) -> list:
        """Convert batch eval examples into batch tensor(custom).

        Can be overwriten, but with the same input parameters and output type.

        @examples(InputFeature): /
        """
        encoder_input = pad_sequence([torch.tensor(feature.encoder_input, dtype=torch.long) for feature in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        ori_mask = (encoder_input != self.tokenizer.pad_token_id)
        # offset_mapping = [feature.offset_mapping for feature in features]
        # labels = pad_sequence([torch.tensor(feature.label, dtype=torch.long) for feature in features], batch_first=True, padding_value=-1)

        return [encoder_input, ori_mask, features]


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

        # calculate result score
        score = self.result.get_score()
        self.logger.info(json.dumps(score, indent=2, ensure_ascii=False))

        # return bad case in train-mode
        if self.setting.bad_case:
            self.return_selected_case(type_='dev_bad_case_pos', items=self.result.bad_case_pos, data_type='dev', epoch=0, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_iden', items=self.result.bad_case_iden, data_type='dev', epoch=0, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_det', items=self.result.bad_case_det, data_type='dev', epoch=0, file_type='excel')
            self.return_selected_case(type_='dev_bad_case_FPR', items=self.result.bad_case_FPR, data_type='dev', epoch=0, file_type='excel')

        # return all result
        self.return_selected_case(type_='eval_prediction', items=self.result.all_result, data_type='test', epoch=0, file_type='excel')
    

    def get_result_on_batch(self, batch: tuple):
        """Return batch output logits during eval model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @batch: /
        """
        encoder_input, ori_mask, features = batch
        predicts = self.beam_search(encoder_input, ori_mask, features)
        
        return predicts, features


    def get_loss_on_batch(self, batch: tuple):
        """Return batch loss during training model(custom).

        Can be overwriten, but with the same input parameters and output type.

        @batch: /
        """
        encoder_input, decoder_input, decoder_output, ori_mask, cor_mask, features = batch
        coverage = Variable(torch.zeros(encoder_input.size(0), encoder_input.size(1))).to(self.device)
        outputs = self.model(encoder_input, decoder_input, ori_mask, cor_mask, decoder_output, coverage)
        return outputs


    def init_vars(self, src, src_mask):  
        init_tok = self.cls_id
        if isinstance(self.model, para.DataParallel) or isinstance(self.model, para.DistributedDataParallel):
            e_output = self.model.module.bart.encoder(src, src_mask)
        else:
            e_output = self.model.bart.encoder(src, src_mask)
        
        outputs = torch.LongTensor([[init_tok] for _ in range(src.size(0))]).to(self.device)

        trg_mask = (outputs != self.tokenizer.pad_token_id)

        coverage_0 = Variable(torch.zeros(src.size(0), src.size(1))).to(self.device)
        if isinstance(self.model, para.DataParallel) or isinstance(self.model, para.DistributedDataParallel):
            d_outputs = self.model.module.bart.decoder(input_ids=outputs, attention_mask=trg_mask,
                                                encoder_hidden_states=e_output['last_hidden_state'], encoder_attention_mask=src_mask)
            ht_hat, attn_dist, _, coverage_step = self.model.module.attention_network(
                                                                decoder_outputs=d_outputs['last_hidden_state'], 
                                                                decoder_input_mask=trg_mask,
                                                                encoder_outputs=e_output['last_hidden_state'], 
                                                                enc_padding_mask=src_mask, 
                                                                coverage=coverage_0)
            p_gen_input = torch.cat((ht_hat, d_outputs['last_hidden_state'], self.model.module.bart.shared(outputs)), -1) 
            p_gen = self.model.module.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)
            output = torch.cat((d_outputs['last_hidden_state'], ht_hat), -1) # B x hidden_dim * 2
            output = self.model.module.out1(output) # B x hidden_dim
            output = self.model.module.out2(output) # B x vocab_size
        else:
            d_outputs = self.model.bart.decoder(input_ids=outputs, attention_mask=trg_mask,
                                                encoder_hidden_states=e_output['last_hidden_state'], encoder_attention_mask=src_mask)
            ht_hat, attn_dist, _, coverage_step = self.model.attention_network(
                                                                decoder_outputs=d_outputs['last_hidden_state'], 
                                                                decoder_input_mask=trg_mask,
                                                                encoder_outputs=e_output['last_hidden_state'], 
                                                                enc_padding_mask=src_mask, 
                                                                coverage=coverage_0)
            p_gen_input = torch.cat((ht_hat, d_outputs['last_hidden_state'], self.model.bart.shared(outputs)), -1) 
            p_gen = self.model.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)
            output = torch.cat((d_outputs['last_hidden_state'], ht_hat), -1) # B x hidden_dim * 3
            output = self.model.out1(output) # B x hidden_dim
            output = self.model.out2(output) # B x vocab_size
        
        vocab_dist = F.softmax(output, dim=-1)
        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist
        encoder_input_ = src.unsqueeze(1).expand(src.size(0), d_outputs['last_hidden_state'].size(1), src.size(1))
        final_dist = vocab_dist_.scatter_add(2, encoder_input_, attn_dist_)
        
        # 忽略S和C，计算所有batch的
        probs, ix = final_dist[:, -1].data.topk(self.setting.beam_k)
            
        # log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
        log_scores = torch.Tensor([[math.log(prob) for prob in probs.data[i]] for i in range(probs.size(0))])
        
        outputs = torch.zeros(src.size(0), self.setting.beam_k, self.setting.max_seq_len).long().to(self.device)
        # if opt.device == 0:
        #     outputs = outputs.cuda()
        outputs[:, :, 0] = init_tok
        # 先给所有batch赋上Generate的token输出
        outputs[:, :, 1] = ix
        
        e_outputs = e_output['last_hidden_state'].unsqueeze(1).expand(src.size(0), self.setting.beam_k, e_output['last_hidden_state'].size(-2), e_output['last_hidden_state'].size(-1))
        
        return outputs, e_outputs, log_scores, coverage_step, attn_dist
        

    def k_best_outputs(self, outputs, out, log_scores, i, k):
        out = out.contiguous().view(out.shape[0]//k, k, out.size(-2), out.size(-1)) # (batch, beam_k, seq_len, vocab_size)
        probs, ix = out[:, :, -1].data.topk(k) # 取seq最后字符输出 (batch, seqlen, topk)
        log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(probs.size(0), k, -1) + log_scores.unsqueeze(-1).expand(log_scores.size(0), log_scores.size(1), k)
        k_probs, k_ix = log_probs.view(log_probs.size(0), -1).topk(k)
        
        row = k_ix // k
        col = k_ix % k

        outputs = outputs.contiguous().view(outputs.shape[0]//k, k, outputs.shape[-1])
        for b in range(outputs.size(0)):
            outputs[b][:, :i] = outputs[b][row[b], :i]
        # outputs[:, :i] = outputs[row, :i]
            outputs[b][:, i] = ix[b][row[b], col[b]]

        # log_scores = k_probs.unsqueeze(0)
        
        return outputs, log_scores


    def beam_search(self, src, src_mask, features):
        batch_outputs = ['' for _ in range(src.shape[0])]
        batch_attn_dist = []
        outputs, e_outputs, log_scores, coverage_step, attn_dist_init = self.init_vars(src, src_mask)
        attn_dist_init = attn_dist_init.unsqueeze(1).expand(attn_dist_init.size(0), self.setting.beam_k, attn_dist_init.size(1), attn_dist_init.size(2))
        attn_dist_init = attn_dist_init.contiguous().view(attn_dist_init.size(0)*self.setting.beam_k, attn_dist_init.size(2), attn_dist_init.size(3))
        batch_attn_dist.append(attn_dist_init)
        # eos_tok = self.tokenizer.eos_id()
        # src_mask = (src != tokenizer.pad_id()).unsqueeze(-2)
        # for i, feature in enumerate(features):
        #     src[i][feature.ori_len:] = tokenizer.unk_id()
        e_outputs = e_outputs.contiguous().view(e_outputs.size(0)*self.setting.beam_k, e_outputs.size(-2), e_outputs.size(-1))
        src = src.unsqueeze(-2).expand(src.size(0), self.setting.beam_k, src.size(-1))
        src = src.contiguous().view(src.size(0)*self.setting.beam_k, src.size(-1))
        src_mask = src_mask.unsqueeze(-2).expand(src_mask.size(0), self.setting.beam_k, src_mask.size(-1))
        src_mask = src_mask.contiguous().view(src_mask.size(0)*self.setting.beam_k, src_mask.size(-1))
        coverage_step = coverage_step.unsqueeze(-2).expand(coverage_step.size(0), self.setting.beam_k, coverage_step.size(-1))
        coverage_step = coverage_step.contiguous().view(coverage_step.size(0)*self.setting.beam_k, coverage_step.size(-1))
        ind = None
        # x_hat的索引矩阵
        # index_x = torch.arange(src.size(0)).view(src.size(0))
        # index_y = torch.zeros(src.size(0), dtype=torch.long)
        for i in range(2, features[0].max_seq_len):
            # trg_mask = self.nopeak_mask(i, batch=src.size(0))
            trg_mask = (outputs != self.tokenizer.pad_token_id)
            # trg_mask = trg_mask.unsqueeze(1).expand(trg_mask.size(0), self.setting.beam_k, trg_mask.size(-2), trg_mask.size(-1)).to(self.device)
            # 降维度
            trg_mask = trg_mask.view(trg_mask.size(0)*self.setting.beam_k, trg_mask.size(-1))
            outputs = outputs.view(outputs.size(0)*self.setting.beam_k, outputs.size(-1))
            # src_mask = src_mask.contiguous().view(src_mask.size(0)*self.setting.beam_k, src_mask.size(-1)).unsqueeze(1)
            # e_outputs = e_outputs.contiguous().view(e_outputs.size(0)*self.setting.beam_k, e_outputs.size(-2), e_outputs.size(-1))

            if isinstance(self.model, para.DataParallel) or isinstance(self.model, para.DistributedDataParallel):
                d_outputs = self.model.module.bart.decoder(input_ids=outputs[:, :i], attention_mask=trg_mask[:, :i],
                                                encoder_hidden_states=e_outputs, encoder_attention_mask=src_mask)
                ht_hat, attn_dist, _, coverage_step = self.model.module.attention_network(
                                                                    decoder_outputs=d_outputs['last_hidden_state'][:, -1].unsqueeze(1), 
                                                                    decoder_input_mask=trg_mask[:, i-1].unsqueeze(-1),
                                                                    encoder_outputs=e_outputs, 
                                                                    enc_padding_mask=src_mask, 
                                                                    coverage=coverage_step)
                p_gen_input = torch.cat((ht_hat, d_outputs['last_hidden_state'][:, -1].unsqueeze(1), self.model.module.bart.shared(outputs[:, i-1].unsqueeze(1))), -1) 
                p_gen = self.model.module.p_gen_linear(p_gen_input)
                p_gen = torch.sigmoid(p_gen)
                output = torch.cat((d_outputs['last_hidden_state'][:, -1].unsqueeze(1), ht_hat), -1) # B x hidden_dim * 2
                output = self.model.module.out1(output) # B x hidden_dim
                output = self.model.module.out2(output) # B x vocab_size
            else:
                d_outputs = self.model.bart.decoder(input_ids=outputs[:, :i], attention_mask=trg_mask[:, :i],
                                                    encoder_hidden_states=e_outputs, encoder_attention_mask=src_mask)
                ht_hat, attn_dist, _, coverage_step = self.model.attention_network(
                                                                    decoder_outputs=d_outputs['last_hidden_state'][:, -1].unsqueeze(1), 
                                                                    decoder_input_mask=trg_mask[:, i-1].unsqueeze(-1),
                                                                    encoder_outputs=e_outputs, 
                                                                    enc_padding_mask=src_mask, 
                                                                    coverage=coverage_step)
                p_gen_input = torch.cat((ht_hat, d_outputs['last_hidden_state'][:, -1].unsqueeze(1), self.model.bart.shared(outputs[:, i-1].unsqueeze(1))), -1) 
                p_gen = self.model.p_gen_linear(p_gen_input)
                p_gen = torch.sigmoid(p_gen)
                output = torch.cat((d_outputs['last_hidden_state'][:, -1].unsqueeze(1), ht_hat), -1) # B x hidden_dim * 3
                output = self.model.out1(output) # B x hidden_dim
                output = self.model.out2(output) # B x vocab_size
            
            # 把每个step中decoder的输出在输入序列的注意力分布存起来
            batch_attn_dist.append(attn_dist)
            assert len(batch_attn_dist) == i

            vocab_dist = F.softmax(output, dim=-1)
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist
            encoder_input_ = src.unsqueeze(1)
            final_dist = vocab_dist_.scatter_add(2, encoder_input_, attn_dist_)

        
            outputs, log_scores = self.k_best_outputs(outputs, final_dist, log_scores, i, self.setting.beam_k)
            
            ones = (outputs==self.sep_id).nonzero() # 获取输出为102的索引矩阵
            sentence_lengths = torch.zeros([outputs.shape[0], outputs.shape[1]], dtype=torch.long).cuda()
            for vec in ones:
                x = vec[0]
                y = vec[1]
                if sentence_lengths[x][y]==0: # First end symbol has not been found yet
                    sentence_lengths[x][y] = vec[-1] # Position of first end symbol

            num_finished_sentences = [len([s for s in sentence_lengths[_] if s > 0]) for _ in range(sentence_lengths.shape[0])]

            # if all(num_finished_sentence == self.setting.beam_k for num_finished_sentence in num_finished_sentences):
                # alpha = 0.7
                # div = 1/(sentence_lengths.type_as(log_scores)**alpha)
                # _, ind = torch.max(log_scores * div, 1)
                # ind = ind.data[0]
                # break
            if any(num_finished_sentence == self.setting.beam_k for num_finished_sentence in num_finished_sentences):
                alpha = 0.7
                for index in range(len(num_finished_sentences)):
                    if num_finished_sentences[index]==self.setting.beam_k and batch_outputs[index]=='':
                        div = 1/(sentence_lengths[index].type_as(log_scores[index])**alpha)
                        _, ind = torch.max(log_scores[index] * div, -1)
                        ind = ind.item()
                        length = (outputs[index][ind]==self.sep_id).nonzero()[0].item()
                        batch_outputs[index] = self.decode(
                                                    features=features,
                                                    output=outputs,
                                                    batch_attn_dist=batch_attn_dist,
                                                    index=index,
                                                    ind=ind,
                                                    length=length)
                if all(num_finished_sentence == self.setting.beam_k for num_finished_sentence in num_finished_sentences):
                    break
        
        for index in range(len(num_finished_sentences)):
            if num_finished_sentences[index] != self.setting.beam_k and batch_outputs[index]=='':
                try:
                    length = (outputs[index][0]==self.sep_id).nonzero()[0].item()
                    batch_outputs[index] = self.decode(
                                                features=features,
                                                output=outputs,
                                                batch_attn_dist=batch_attn_dist,
                                                index=index,
                                                ind=ind,
                                                length=length)
                except:
                    batch_outputs[index] = ' '

        return batch_outputs


    def decode(self, features, output, batch_attn_dist, index, ind, length):
        # 解码，处理[UNK]问题
        # try:
        #     assert len(batch_attn_dist) == length+1
        # except:
        #     print(len(batch_attn_dist), length+1)
        batch_attn_dist_ = torch.cat(batch_attn_dist, dim=1) # batch*beam-k, decoder_len, encoder_len
        batch_attn_dist_ = batch_attn_dist_.contiguous().view(output.size(0), output.size(1), batch_attn_dist_.size(1), batch_attn_dist_.size(2))
        pred = self.tokenizer.decode(output[index][ind][1:length]).split()
        for i, char in enumerate(pred):
            if char == '[UNK]':
                position = features[index].offsets_mapping[torch.argmax(batch_attn_dist_[index][ind][i])]
                pred[i] = features[index].text[position[0]:position[1]]

        return ' '.join(pred)




