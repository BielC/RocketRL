import rlgym
import time
import numpy as np
import ppo
import datetime
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.reward_functions.common_rewards \
    import PlayerTowardsBallReward, GoalReward, BallTowardsGoalReward
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition
from torch.utils.tensorboard import SummaryWriter

# ENV PARAMETERS
ep_len_secs = 25

tick_skip = 8

physics_ticks_per_sec = 120

# AGENT PARAMETERS
N_GAMES = 100000
N = 512
BATCH_SIZE = 256
N_EPOCHS = 10
ACTOR_LR = 2e-6
CRITIC_LR = 4e-5
CLIP_RANGE = 0.1
GAMMA = 0.99
LAMBDA = 0.95

max_steps = int(round(ep_len_secs * physics_ticks_per_sec / tick_skip))


# Set log directory
logdir = "runs/agent_%s_N%d_B%d_A%.1e_C%.1e" % (datetime.datetime.now().strftime("%m%d-%H%M"),
                                                N, BATCH_SIZE, ACTOR_LR, CRITIC_LR)

# Init tensorboard writer
writer = SummaryWriter(log_dir=logdir, flush_secs=60)


#vSet custom reward
reward = CombinedReward(reward_functions=(PlayerTowardsBallReward(),
                                          BallTowardsGoalReward(),
                                          GoalReward()),
                        reward_weights=(1, 5, 20))

# Init environment and launch RL
env = rlgym.make("default",
                 spawn_opponents=False,
                 reward_fn=PlayerTowardsBallReward(),
                 obs_builder=AdvancedObs(),
                 terminal_conditions=[TimeoutCondition(max_steps), GoalScoredCondition(), ],
                 game_speed=50,
                 tick_skip=8)

# Init agent
agent = ppo.Agent(n_actions=8,
                  batch_size=BATCH_SIZE,
                  n_epochs=N_EPOCHS,
                  input_dims=env.observation_space.shape,
                  writer=writer,
                  actor_lr=ACTOR_LR,
                  critic_lr=CRITIC_LR,
                  gamma=GAMMA,
                  glambda=LAMBDA,
                  policy_clip=CLIP_RANGE
                  )


best_score = env.reward_range[0]
score_history = []
Tsteps = 0
for i in range(N_GAMES):

    obs = env.reset()
    done = False
    score = 0
    steps = 0
    learning_steps = 0
    avg_score = 0
    t0 = time.time()
    while not done:
        #Get action from agent
        action, prob, value = agent.act(obs)

        #Log actions histogram
        [writer.add_histogram("Actions", action[i], i) for i in range(8)]

        #Run env with action
        new_obs, reward, done, info = env.step(action)

        score += reward
        steps += 1
        Tsteps += 1

        #Store step to buffer
        agent.remember(obs, action, prob, value, reward, done)

        # Learn every batch steps
        if Tsteps % N == 0:
            print("Learning...")
            try:
                agent.learn()
            except KeyboardInterrupt:
                print("Stopped: Saving models...")
                agent.save_models(i)
            learning_steps += 1

        #store observations for next step
        obs = new_obs

    score_history.append(score)
    avg_score = np.mean(score_history[-50:])

    agent.w.add_scalar('Episode/ Avg. Score', avg_score, i)
    agent.w.add_scalar('Episode/ Score', score, i)

    #Save models if it's better than average or every 100 steps
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models(i)
        writer.add_scalar('Episode/Update', 1, i)
    elif i % 100 == 0:
        agent.save_models(i)
        writer.add_scalar('Episode/Update', 0, i)
    else:
        writer.add_scalar('Episode/Update', 0, i)

    length = time.time() - t0
    print("Ep %d. Step time: %.3f | Episode time: %.2f | Episode Reward: %.2f"
          % (i, length / steps, length, score))

#
agent.save_models(N_GAMES)
env.close()
writer.close()
