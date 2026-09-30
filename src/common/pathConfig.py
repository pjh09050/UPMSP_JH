import os
class pathConfig:
    absolute_path = ""
    log_path = ""
    model_save_path = ""
    pickle_data_path = ""
    pickle_simulator_data_path = ""
    gantt_save_path = ""
    simulator_result_path = ""
    running_data_path = ""
    os = "/"
    @classmethod
    def set_absolute_path(cls):
        script_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(script_path)
        cls.absolute_path = os.path.dirname(dir_path) # src 파일 경로
        cls.log_path = f"{cls.absolute_path}{os.sep}log_data"
        cls.model_save_path = f"{cls.absolute_path}{os.sep}params_data"
        cls.pickle_data_path = f"{cls.absolute_path}{os.sep}master_db{os.sep}pickleDBData"
        cls.simulator_result_path = f"{cls.absolute_path}{os.sep}simulator_result"
        cls.simulator_result_plot_save_path = f"{cls.absolute_path}{os.sep}simulator_result_plot"
        cls.running_data_path = f"{cls.absolute_path}{os.sep}running_data"
        cls.pickle_simulator_data_path = f"{cls.absolute_path}{os.sep}master_db{os.sep}pickleSimulatorData"