import os
from easy_task.base_module import BaseUtils, TaskSetting
from ptr_gen_net_task import GrammarCorrectTask


if __name__ == '__main__':
    # init task utils
    task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'config/grammar_ptr_train.yml'))
    # task_utils = BaseUtils(task_config_path=os.path.join(os.getcwd(), 'config/grammar_ptr_predict.yml'))

    # init task setting
    task_setting = TaskSetting(task_utils.task_configuration)

    # build custom task
    task = GrammarCorrectTask(task_setting, load_train=not task_setting.skip_train, load_dev=not task_setting.skip_train, load_test=True if hasattr(task_setting, 'load_test') and task_setting.load_test else False)

    # do train
    if not task_setting.skip_train:
        task.output_result['result_type'] = 'Train_mode'
        # 从0开始训练
        task.train()
        # 基于已有的模型checkpoint开始二阶段训练
        # task.train(resume_model_path='/home/liyunliang/CGED_Task/output/Grammar_correct_task_ptr-gen_bart-large_5e-5_lang8+linian_cosine_drop0.1_beam3_check/Model/Grammar_correct_task_ptr-gen_bart-large_5e-5_lang8+linian_cosine_drop0.1_beam3_check.cpt.dev.0.e(10).b(16).p(1。0).s(99)')
    # do test
    else:
        task.output_result['result_type'] = 'Test_mode'
        task.logger.info('Skip training')
        task.logger.info('Start evaling')

        # load checkpoint and do eval
        task.resume_test_at(task.setting.resume_model_path)