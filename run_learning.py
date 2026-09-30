from src.learner.common.Hyperparameters import *
from src.learner.Algorithm.DQN_rule_select import *
from src.learner.Algorithm.DQN_job_select import *
from src.learner.Algorithm.MC_job_select import *
from src.learner.Algorithm.MC_rule_select import *
from src.learner.Algorithm.Two_Stage_DQN import *
from src.learner.Algorithm.Two_Stage_MC_DQN import *
from src.learner.Algorithm.Two_Stage_MC_DQN_2 import *
from src.learner.Algorithm.PMS_PPO_batch import *
from src.learner.Algorithm.Two_stage_RNN_td import *
from src.learner.Algorithm.Two_stage_RNN_mc import *
import yaml
np.set_printoptions(linewidth=200)

class Run_Simulator:
    def __init__(self, dataset, mode):
        Parameters.set_time_to_string()  # 현재 시간 가져오는 코드 -> 로그 및 기록을 위함
        Parameters.set_absolute_path()
        Parameters.set_dataSetId([dataset])
        Parameters.set_mode(mode)
        with open(f'{pathConfig.absolute_path}{os.sep}hyperparameter.yaml', 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        Parameters.init_parameter_setting(config_data['engine'])
        Parameters.set_plan_horizon(100000)
        Hyperparameters.init_hyperparameter_setting(config_data['hyperparameter'])  # 시뮬레이터의 하이퍼파라미터 설정
        action_list = []
        for i in range(Hyperparameters.family_type):
            action_list.append(f"{i + 1}_SPTSDR")
            action_list.append(f"{i + 1}_SPTLDR")
            action_list.append(f"{i + 1}_LPTSDR")
            action_list.append(f"{i + 1}_LPTLDR")
        print(f"action_list: {action_list}")
        Simulator._init(Parameters.datasetId)  # 데이터셋 선택 후 데이터에 맞게 시뮬레이터 설정
        Hyperparameters.init_rl_config_setting(config_data['configRL'], action_list, Simulator)  # rl 요소 설정
        print("set complete")
    def main(self, mode, algorithm, dataset, folder_name, param):
        logging.info(f"mode: {mode}")
        logging.info(f"dsp_rule: {algorithm}")
        if mode == "learning":
            if algorithm == 'DQN':
                if Hyperparameters.action_type == "job_select" or Hyperparameters.action_type == "family_type" or Hyperparameters.action_type == "continuous_select":
                    DQN_Job_Select.main()
                elif Hyperparameters.action_type == 'dsp_rule':
                    makespan = DQN_Rule_Select.main()
            elif algorithm == 'MC':
                if Hyperparameters.action_type == "mc_job_select" or Hyperparameters.action_type == "family_type":
                    MC_Job_Select.main()
                elif Hyperparameters.action_type == 'dsp_rule':
                    makespan = MC_Rule_Select.main()
            elif algorithm == 'Two_Stage_DQN':
                if Hyperparameters.action_type == 'two_stage_dqn' or Hyperparameters.action_type == 'two_stage_dqn_rule':
                    Two_Stage_DQN.main()
            elif algorithm == 'Two_Stage_MC_DQN':
                if Hyperparameters.action_type == 'two_stage_dqn' or Hyperparameters.action_type == 'two_stage_dqn_rule':
                    makespan = Two_Stage_MC_DQN.main()
            elif algorithm == "Two_Stage_RNN":
                if Hyperparameters.action_type == "two_stage_rnn":
                    # Two_Stage_RNN_td.main()
                    makespan = Two_Stage_RNN_MC.main()
            elif algorithm == 'PPO':
                if Hyperparameters.action_type == "ppo_job_select" or Hyperparameters.action_type == "ppo_family_type":
                    PMS_PPO_batch.main()
        elif mode == 'evaluate':
            if algorithm == "DQN":
                if Hyperparameters.action_type == "job_select" or Hyperparameters.action_type == "family_type":
                    DQN_Job_Select.get_evaluate(f"{pathConfig.model_save_path}{os.sep}{folder_name}", 3000, 1500,
                                                [dataset])
                elif Hyperparameters.action_type == 'dsp_rule':
                    DQN_Rule_Select.get_evaluate(f"{pathConfig.model_save_path}{os.sep}{folder_name}", 4000, 4000,
                                                 [dataset])
            elif algorithm == "Two_Stage_MC_DQN":
                Two_Stage_MC_DQN_2.get_evaluate(f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent1",
                                              f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent2", [dataset])
            elif algorithm == 'MC':
                if Hyperparameters.action_type == "mc_job_select" or Hyperparameters.action_type == "family_type":
                    MC_Job_Select.get_evaluate(f"{pathConfig.model_save_path}{os.sep}{folder_name}", 2000, 1500,
                                               [dataset])
                elif Hyperparameters.action_type == 'dsp_rule':
                    MC_Rule_Select.get_evaluate(f"{pathConfig.model_save_path}{os.sep}{folder_name}", 3000, 1500,
                                                [dataset])
            elif algorithm == 'PPO':
                if Hyperparameters.action_type == "ppo_job_select" or Hyperparameters.action_type == "ppo_family_type":
                    PMS_PPO_batch.get_evaluate(f"{pathConfig.model_save_path}{os.sep}{folder_name}", 3000, 1000,
                                                [dataset])
        elif mode == "result":
            if algorithm == "DQN":
                if Hyperparameters.action_type == "job_select" or Hyperparameters.action_type == "family_type":
                    DQN_Job_Select.get_result(
                        f"{pathConfig.model_save_path}{os.sep}{folder_name}{os.sep}{param}param.pt", [dataset])
                elif Hyperparameters.action_type == 'dsp_rule':
                    DQN_Rule_Select.get_result(
                        f"{pathConfig.model_save_path}{os.sep}{folder_name}{os.sep}{param}param.pt", [dataset])
            elif algorithm == 'Two_Stage_DQN':
                if Hyperparameters.action_type == 'two_stage_dqn' or Hyperparameters.action_type == 'two_stage_dqn_rule':
                    Two_Stage_DQN.get_result(f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent1",
                                             f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent2", [dataset])
            elif algorithm == "Two_Stage_MC_DQN":
                if Hyperparameters.action_type == 'two_stage_dqn' or Hyperparameters.action_type == 'two_stage_dqn_rule':
                    Two_Stage_MC_DQN_2.get_result(f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent1",
                                                f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent2", [dataset],
                                                param)
            elif algorithm == 'MC':
                if Hyperparameters.action_type == "mc_job_select" or Hyperparameters.action_type == "family_type":
                    MC_Job_Select.get_result(
                        f"{pathConfig.model_save_path}{os.sep}{folder_name}{os.sep}{param}param.pt", [dataset])
                elif Hyperparameters.action_type == 'dsp_rule':
                    MC_Rule_Select.get_result(
                        f"{pathConfig.model_save_path}{os.sep}{folder_name}{os.sep}{param}param.pt", [dataset])
            elif algorithm == 'PPO':
                if Hyperparameters.action_type == "ppo_job_select" or Hyperparameters.action_type == "ppo_family_type":
                    PMS_PPO_batch.get_result(
                        f"{pathConfig.model_save_path}{os.sep}{folder_name}{os.sep}{param}_policy_param.pt",
                        f"{pathConfig.model_save_path}{os.sep}{folder_name}{os.sep}{param}_value_param.pt", [dataset])
        elif mode == "fix_agent1":
            if algorithm == 'Two_Stage_MC_DQN':
                if Hyperparameters.action_type == 'two_stage_dqn' or Hyperparameters.action_type == 'two_stage_dqn_rule':
                    makespan = Two_Stage_MC_DQN_2.fix_agent1_main(
                        f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent1")
        elif mode == "train_agent1":
            if algorithm == 'Two_Stage_MC_DQN':
                if Hyperparameters.action_type == 'two_stage_dqn' or Hyperparameters.action_type == 'two_stage_dqn_rule':
                    save_directory = Two_Stage_MC_DQN_2.only_train_agent1()
                    return save_directory
        elif mode == "inference":
            if algorithm == "Two_Stage_MC_DQN":
                if Hyperparameters.action_type == 'two_stage_dqn' or Hyperparameters.action_type == 'two_stage_dqn_rule':
                    agent1_directory = f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent1"
                    agent2_directory = f"{pathConfig.model_save_path}{os.sep}{folder_name}_agent2"
                    print("=" * 80)
                    print("[Trained Policy Inference]")
                    print(f"New dataset : {dataset}")
                    print(f"Agent1 dir  : {agent1_directory}")
                    print(f"Agent2 dir  : {agent2_directory}")
                    print("=" * 80)
                    makespan = Two_Stage_MC_DQN.run_trained_policy(
                        new_dataset_ids=[dataset],
                        agent1_directory=agent1_directory,
                        agent2_directory=agent2_directory,
                        plot_save_directory=f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
            )
            elif algorithm == "Two_Stage_RNN":
                if Hyperparameters.action_type == "two_stage_rnn":
                    model_directory = (f"{pathConfig.model_save_path}" f"{os.sep}{folder_name}")
                    makespan = Two_Stage_RNN_MC.run_trained_policy(
                        new_dataset_ids=[dataset], model_directory=model_directory, param=param
                    )
        return makespan

if True:
    # dataset = "M10_J200_F3_P10-50_D5-10_Fgap_U1" # instance 1
    dataset = "M10_J500_F5_P10-50_D8-13_Fgap_U3" # instance 12
    # dataset = "M25_J500_F5_P10-50_D5-10_Fgap_U1" # instance 13
    # dataset = "M10_J500_F7_P10-50_D8-13_Fgap_U3" # instance 20
    
    mode = "inference"
    Algorithm = "Two_Stage_MC_DQN"
    folder_name = "260917_111804_['M10_J500_F5_P10-50_D5-10_Fgap_U1']"  # 뒤에 에이전트 지우고 작성
    param = None  # two-step-mc-dqn은 굳이 설정 안해도 best 파라미터 탐색해서 결과 보여줌
    makespan_list = []

    total_start_time = time.time()
    for i in range(1):
        start_time = time.time()
        simulator = Run_Simulator(dataset, mode)
        makespan = simulator.main(mode, Algorithm, dataset, folder_name, param)
        makespan_list.append(makespan)
        one_track_time = (time.time() - start_time) / 60
        print(f"learning (소요 시간: {one_track_time:.2f}분)")
    elapsed_min = (time.time() - total_start_time) / 60
    makespan_array = np.array(makespan_list)
    mean_makespan = np.mean(makespan_array)
    std_makespan = np.std(makespan_array)
    print(f"평균 Makespan: {mean_makespan:.2f}")
    print(f"표준편차: {std_makespan:.2f}")
    print(f"learning (소요 시간: {elapsed_min:.2f}분)")