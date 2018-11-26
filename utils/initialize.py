from constants import constants
from environment.reacher import ReacherEnv
from agent.agent import Agent


def initialize(
    environment_path,
    seed=constants.SEED,
    gamma=constants.GAMMA,
    tau=constants.TAU,
    critic_lr=constants.CRITIC_LR,
    actor_lr=constants.ACTOR_LR,
    buffer_size=constants.BUFFER_SIZE,
    batch_size=constants.BATCH_SIZE,
    actor_layer_1_nodes=constants.FC1_UNITS,
    critic_layer_1_nodes=constants.FC2_UNITS,
    actor_layer_2_nodes=constants.FC1_UNITS,
    critic_layer_2_nodes=constants.FC2_UNITS,
    actor_model_path=None,
    critic_model_path=None
):
    reacherEnv = ReacherEnv(environment_path)
    action_size = reacherEnv.get_action_size()
    state_size = reacherEnv.get_state_size()
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        random_seed=seed,
        gamma=gamma,
        tau=tau,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        actor_layer_1_nodes=actor_layer_1_nodes,
        critic_layer_1_nodes=critic_layer_1_nodes,
        actor_layer_2_nodes=actor_layer_2_nodes,
        critic_layer_2_nodes=critic_layer_2_nodes
    )
    if actor_model_path and critic_model_path:
        print('Loaded weights')
        agent.load_saved_actor_model(actor_model_path) \
            .load_saved_critic_model(critic_model_path)
    return reacherEnv, agent
