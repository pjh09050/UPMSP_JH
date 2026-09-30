from src.common.Parameters import *
from src.master_db.DataInventory import *
class Hyperparameters:
    gamma = 0.99
    learning_rate = 0.0001
    batch_size = 32
    buffer_limit = 50000
    input_layer = 51
    output_layer = 10
    episode = 2000
    end_type = ""
    end_time = 300
    family_type = 5
    family_setup = ""
    maintenance_time = 30
    max_job = 3
    T = 0
    R = 0
    object = ""

    reward_type = ""
    state_type = ""
    action_type = ""
    maintenance = False
    donothing = False

    action_count = ""
    action_dimension = ""

    @classmethod
    def init_hyperparameter_setting(cls,config):
        cls.gamma = config['gamma']
        cls.learning_rate = config['learning_rate']
        cls.batch_size = config['batch_size']
        cls.buffer_limit = config['buffer_limit']
        cls.input_layer = config['input_layer']
        cls.output_layer = config['output_layer']
        cls.episode = config['episode']
        cls.initial_setup = config['initial_setup']
        cls.job_arrival_end_type = config['job_arrival_end_type']
        cls.end_time = config['end_time']
        cls.family_type = config['family_type']
        cls.family_setup = config['family_setup']
        cls.maintenance_time = config['maintenance_time']
        cls.max_job = config['max_job']
        cls.T = config['T']
        cls.R = config['R']
        cls.object = config['object']
        cls.maintenance_threshold = config['maintenance_threshold']

    @classmethod
    def init_rl_config_setting(cls, config, action_list, simulator):
        cls.state_type = config['state_type']
        cls.reward_type = config['reward_type']
        cls.maintenance = config['maintenance']
        cls.donothing = config['donothing']
        cls.action_type = config['action_type']
        cls.set_action_type(cls.action_type, action_list)
        cls.set_state_dimension(simulator)

    @classmethod
    def set_action_type(cls, action_type, action_list):
        if cls.action_type == "two_stage_dqn":
            cls.action_count = Hyperparameters.family_type
        elif action_type == "dsp_rule":
            if Hyperparameters.maintenance == True:
                cls.rule_count = int(len(action_list) / Hyperparameters.family_type)
                cls.action_type = action_type
                action_list.append("Maintenance")
                cls.action_count = len(action_list)
                cls.action_dimension = action_list
        elif action_type == "Family_LDR" or action_type == "two_stage_dqn_rule":
            cls.action_type = action_type
            cls.action_count = len(action_list)
            cls.action_dimension = action_list
            cls.rule_count = int(len(action_list) / Hyperparameters.family_type)
        else:
            cls.action_type = action_type
            DataInventory.set_db_data(Parameters.datasetId[0])
            if Hyperparameters.maintenance == True:
                if Hyperparameters.donothing == True:
                    job_types = [job.jobType for job in DataInventory.master_data["Job_db"]]
                    job_types_sorted = sorted(job_types, key=lambda x: int(x[1:]))  # 'J01' → 1
                    cls.action_dimension = job_types_sorted + ['Maintenance', 'Nothing']
                else:
                    job_types = [job.jobType for job in DataInventory.master_data["Job_db"]]
                    job_types_sorted = sorted(job_types, key=lambda x: int(x[1:]))  # 'J01' → 1
                    cls.action_dimension = job_types_sorted + ['Maintenance']
            else:
                if Hyperparameters.donothing == True:
                    job_types = [job.jobType for job in DataInventory.master_data["Job_db"]]
                    job_types_sorted = sorted(job_types, key=lambda x: int(x[1:]))  # 'J01' → 1
                    cls.action_dimension = job_types_sorted + ['Nothing']
                else:
                    job_types = [job.jobType for job in DataInventory.master_data["Job_db"]]
                    job_types_sorted = sorted(job_types, key=lambda x: int(x[1:]))  # 'J01' → 1
                    cls.action_dimension = job_types_sorted
            cls.action_count = len(Hyperparameters.action_dimension)
        cls.output_layer = cls.action_count
        print('output_layer', cls.output_layer)

    @classmethod
    def set_reward_type(cls, reward_type):
        cls.reward_type = reward_type

    @classmethod
    def set_state_type(cls, state):
        cls.state_type = state

    @classmethod
    def set_state_dimension(cls, Simulator):
        if cls.state_type == "MC_DQN_state":
            cls.input_layer = Hyperparameters.family_type * 4 + 6
        elif cls.state_type == "state_36":
            cls.input_layer = Simulator.number_of_machine * 3 + 6
        elif cls.state_type == "two_stage_rnn":
            cls.input_layer = 6
        print('input_layer', cls.input_layer)