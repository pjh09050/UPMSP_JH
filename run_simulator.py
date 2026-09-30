from src.common.Parameters import *
import yaml
from src.simlator.Simulator import *

class Run_Simulator:
    def __init__(self, data):
        print("simulator on")
        Parameters.set_time_to_string() # 현재 시간 가져오는 코드 -> 로그 및 기록을 위함
        Parameters.set_absolute_path()

        with open(f'{pathConfig.absolute_path}/hyperparameter.yaml', 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        Parameters.init_parameter_setting(config_data['engine'])
        Parameters.set_dataSetId([data]) # instance 1

        action_list = ["Family_LDR"]
        Hyperparameters.init_hyperparameter_setting(config_data['hyperparameter'])  # 시뮬레이터의 하이퍼파라미터 설정
        Simulator.init_simulator(Parameters.datasetId) # 데이터셋 선택 후 데이터에 맞게 시뮬레이터 설정
        Hyperparameters.init_rl_config_setting(config_data['configRL'], action_list, Simulator)  # rl 요소 설정
        print("set complete")

    def main(self, mode, dsp_rule):
        logging.info(f"mode: {mode}")
        logging.info(f"dsp_rule: {dsp_rule}")
        if mode == "DSP_run":
            Simulator.run(dsp_rule)

if True:
    # dataset = "M10_J200_F3_P10-50_D5-10_Fgap_U1"  # instance 1
    # dataset = "M10_J500_F5_P10-50_D5-10_Fgap_U1" # instance 12
    # dataset = "M25_J500_F5_P10-50_D5-10_Fgap_U1" # instance 13
    dataset = "M10_J500_F7_P10-50_D5-10_Fgap_U1" # instance 20

    time_list = []
    for i in range(1):
        start_time = time.time()
        simulator = Run_Simulator(dataset)
        # 사용 가능 DSP rule = SPT, LPT, SPTSSU, SSU, LOR, MOR, EDD, MST, FIFO, LIFO, SETUP_LPT, Family_LPT
        simulator.main("DSP_run", "Family_LDR")  # dsp_rule = 개별 확인할 때만 사용하면 됨
        elapsed_min = (time.time() - start_time)
        print(f"one episode (소요 시간: {elapsed_min:6f})")
        time_list.append(elapsed_min)

    # mean_time_list = np.mean(time_list)
    # std_time_list = np.std(time_list)
    #
    # txt_name = (
    #     f"{Parameters.simulation_time}_"
    #     f"{Parameters.datasetId}_"
    #     f"F{Hyperparameters.family_type}_"
    #     f"rule_result.txt"
    # )
    #
    # with open(txt_name, "w", encoding="utf-8") as f:
    #     f.write("===== Rule Result =====\n")
    #     f.write(f"Evaluation Time list (sec): {time_list:}\n")
    #     f.write(f"mean time list : {mean_time_list:}\n")
    #     f.write(f"std time list : {std_time_list}\n")
    # print(f"📄 Best result TXT 저장 완료: {txt_name}")

    # dataset = "M10_J500_F5_P10-50_D8-13_Fgap_U2" # instance 12
    # time_list = []
    # for i in range(1):
    #     start_time = time.time()
    #     simulator = Run_Simulator(dataset)
    #     simulator.main("DSP_run", "Family_LDR")  # dsp_rule = 개별 확인할 때만 사용하면 됨
    #     elapsed_min = (time.time() - start_time)
    #     print(f"one episode (소요 시간: {elapsed_min:6f})")
    #     time_list.append(elapsed_min)
    #
    # dataset = "M10_J500_F5_P10-50_D8-13_Fgap_U3" # instance 12
    # time_list = []
    # for i in range(1):
    #     start_time = time.time()
    #     simulator = Run_Simulator(dataset)
    #     simulator.main("DSP_run", "Family_LDR")  # dsp_rule = 개별 확인할 때만 사용하면 됨
    #     elapsed_min = (time.time() - start_time)
    #     print(f"one episode (소요 시간: {elapsed_min:6f})")
    #     time_list.append(elapsed_min)