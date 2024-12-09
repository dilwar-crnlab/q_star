
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym
from IPython.display import clear_output
import torch

import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn.policies import MlpPolicy

class PathSelectionWrapper(gym.Wrapper):
    """Wrapper for path selection phase"""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space['path_action']
        self.observation_space = env.observation_space['path_obs']
        
    def step(self, action):
        # Wrap action in dict as expected by environment
        full_action = {'path_action': action, 'spectrum_action': 0}
        obs, reward, done, info = self.env.step(full_action)
        if isinstance(obs, dict):
            return obs['path_obs'], reward, done, info
        return obs, reward, done, info
        
    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, dict):
            return obs['path_obs']
        return obs

class SpectrumAssignmentWrapper(gym.Wrapper):
    """Wrapper for spectrum assignment phase"""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space['spectrum_action']
        self.observation_space = env.observation_space['spectrum_obs']
        
    def step(self, action):
        # Wrap action in dict as expected by environment
        full_action = {'path_action': 0, 'spectrum_action': action}
        obs, reward, done, info = self.env.step(full_action)
        if isinstance(obs, dict):
            return obs['spectrum_obs'], reward, done, info
        return obs, reward, done, info
        
    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, dict):
            return obs['spectrum_obs']
        return obs

class QStarTrainingCallback(BaseCallback):
    """Callback for saving model and monitoring training"""
    def __init__(self, check_freq: int, log_dir: str, phase: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.phase = phase
        
        # Initialize metrics
        self.rewards = []
        self.success_rates = []
        self.episode_lengths = []
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Load results
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Compute mean reward
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"\n{self.phase} Training step {self.n_calls}")
                    print(f"Mean reward: {mean_reward:.2f}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f}")
                    
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                    
                # Update metrics
                self.rewards.append(mean_reward)
                
                # Plot metrics
                self._plot_training_progress()
                
        return True
    
    def _plot_training_progress(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(131)
        plt.plot(self.rewards)
        plt.title(f'{self.phase} Rewards')
        plt.xlabel('Training step')
        plt.ylabel('Mean reward')
        
        if len(self.success_rates) > 0:
            plt.subplot(132)
            plt.plot(self.success_rates)
            plt.title(f'{self.phase} Success Rate')
            plt.xlabel('Training step')
            plt.ylabel('Success rate')
        
        if len(self.episode_lengths) > 0:
            plt.subplot(133)
            plt.plot(self.episode_lengths)
            plt.title(f'{self.phase} Episode Lengths')
            plt.xlabel('Training step')
            plt.ylabel('Length')
        
        plt.tight_layout()
        plt.show()

def create_dqn_agents(env):
    """Create separate DQN agents for path and spectrum selection"""
    # Create wrapped environments
    path_env = PathSelectionWrapper(env)
    spectrum_env = SpectrumAssignmentWrapper(env)
    
    # Path finding DQN
    path_policy_args = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU
    )
    
    path_dqn = DQN(
        MlpPolicy,
        path_env,
        verbose=0,
        tensorboard_log="./tb/PathDQN/",
        policy_kwargs=path_policy_args,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        target_update_interval=500,
        train_freq=4
    )
    
    # Spectrum assignment DQN
    spectrum_policy_args = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU
    )
    
    spectrum_dqn = DQN(
        MlpPolicy,
        spectrum_env,
        verbose=0,
        tensorboard_log="./tb/SpectrumDQN/",
        policy_kwargs=spectrum_policy_args,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.95,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4
    )
    
    return path_dqn, spectrum_dqn

def train_qstar_deeprmsa(total_timesteps=1000000):
    """Train both path selection and spectrum assignment agents"""
    # Load topology
    topology_name = 'dutch'
    k_paths = 3
    with open(f'../topologies/{topology_name}_{k_paths}-paths_6-modulations.h5', 'rb') as f:
        topology = pickle.load(f)

    # Create environment
    env_args = dict(
        topology=topology,
        seed=10,
        allow_rejection=False,
        mean_service_holding_time=10.0,
        episode_length=50,
        node_request_probabilities=None
    )
    
    env = gym.make('ModDeepRMSA-v0', **env_args)
    
    # Create log directories
    os.makedirs("./tmp/path", exist_ok=True)
    os.makedirs("./tmp/spectrum", exist_ok=True)
    
    # Create wrapped environments with monitoring
    path_env = Monitor(PathSelectionWrapper(env), "./tmp/path/training")
    spectrum_env = Monitor(SpectrumAssignmentWrapper(env), "./tmp/spectrum/training")
    
    # Create callbacks
    path_callback = QStarTrainingCallback(check_freq=1000, log_dir="./tmp/path/", phase="Path")
    spectrum_callback = QStarTrainingCallback(check_freq=1000, log_dir="./tmp/spectrum/", phase="Spectrum")
    
    # Create agents
    path_dqn, spectrum_dqn = create_dqn_agents(env)
    
    print("Starting training...")
    print("\nTraining path selection agent...")
    path_dqn.learn(
        total_timesteps=total_timesteps,
        callback=path_callback,
        tb_log_name="path_selection"
    )
    
    print("\nTraining spectrum assignment agent...")
    spectrum_dqn.learn(
        total_timesteps=total_timesteps,
        callback=spectrum_callback,
        tb_log_name="spectrum_assignment"
    )
    
    # Save final models
    path_dqn.save("./models/path_dqn_final")
    spectrum_dqn.save("./models/spectrum_dqn_final")
    
    return path_dqn, spectrum_dqn, path_callback

if __name__ == "__main__":
    # Register environment if needed
    # gym.register(id='ModDeepRMSA-v0', entry_point='optical_rl_gym.envs:ModifiedDeepRMSAEnv')
    
    # Train agents
    path_dqn, spectrum_dqn, callback = train_qstar_deeprmsa(total_timesteps=1000000)
    
    # Plot final training curves
    callback._plot_training_progress()
