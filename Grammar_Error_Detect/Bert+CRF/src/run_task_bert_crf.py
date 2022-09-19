import os
import sys
sys.path.append('/home/liyunliang/CGED_Task/src/transformer')
from easy_task.base_module import BaseUtils, TaskSetting
from Grammar_Error_Detect.src.bert_crf_task import GrammarDetectTask


if __name__ == '__main__':
    # init task utils
    task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'config/bert_crf_train.yml'))
    # task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'config/bert_crf_predict.yml'))

    # init task setting
    task_setting = TaskSetting(task_utils.task_configuration)

    # build custom task
    task = GrammarDetectTask(task_setting, load_train=not task_setting.skip_train, load_dev=not task_setting.skip_train, load_test=True if hasattr(task_setting, 'load_test') and task_setting.load_test else False)

    # do train
    if not task_setting.skip_train:
        task.output_result['result_type'] = 'Train_mode'
        # 训练集为lang8，则从0开始训练
        if 'lang8' in task_setting.train_file_name:
            task.train()
        # 训练集为历年数据集，则基于lang8的模型继续训练
        else:
            task.train(resume_model_path='/home/liyunliang/CGED_Task/output/Grammar_detect_task_macbert-large_5e-6_lang8-all-base_linian-aug_drop0.3_cosine/Model/Grammar_detect_task_macbert-large_5e-6_lang8-all-base_linian-aug_drop0.3_cosine.cpt.dev.0.e(10).b(32).p(1。0).s(99)')
    # do test
    else:
        task.output_result['result_type'] = 'Test_mode'
        task.logger.info('Skip training')
        task.logger.info('Start evaling')

        # load checkpoint and do eval
        task.resume_test_at(task.setting.resume_model_path)