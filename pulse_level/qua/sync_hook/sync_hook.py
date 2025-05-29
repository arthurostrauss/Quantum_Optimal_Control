from rl_qoc.agent import PPOConfig, TrainingConfig, TrainFunctionSettings
from rl_qoc.qua import QMEnvironment
from iqcc_cloud_client.runtime import get_qm_job

job = get_qm_job()

q_env = QMEnvironment(q_env_config, job=job)
ppo_config = PPOConfig.from_yaml("agent_config.yaml")
ppo_agent = CustomQMPPO(ppo_config, rescaled_env)

ppo_training = TrainingConfig(num_updates)
ppo_settings = TrainFunctionSettings(plot_real_time=True, print_debug=True)
