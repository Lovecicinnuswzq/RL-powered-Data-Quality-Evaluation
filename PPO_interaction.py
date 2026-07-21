import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

from PPO_env import YoloDVRLEnv


class CustomGaussianMlpPolicy(ActorCriticPolicy):
    """
    Custom MLP Policy for RL-evaluator with specific weight initialization
    and log_std configuration for Bernoulli/Gaussian sampling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.action_net, nn.Linear):
            nn.init.zeros_(self.action_net.weight)
            nn.init.constant_(self.action_net.bias, 0.5)

        with torch.no_grad():
            self.log_std.fill_(-2)  # log(0.2) ≈ -1.6


class PPOInterface:
    def __init__(self,
                 image_ids,
                 base_train_img_dir,
                 base_train_lbl_dir,
                 val_img_dir,
                 val_lbl_dir,
                 model_path="yolo11n.pt",
                 feature_npz_path="features.npz",
                 load_evaluator_path=None,
                 tensorboard_log_dir="runs_dvrl"):

        def make_env():
            return YoloDVRLEnv(
                image_ids=image_ids,
                base_train_img_dir=base_train_img_dir,
                base_train_lbl_dir=base_train_lbl_dir,
                val_img_dir=val_img_dir,
                val_lbl_dir=val_lbl_dir,
                model_path=model_path,
                feature_npz_path=feature_npz_path
            )

        self.env = DummyVecEnv([make_env])

        if load_evaluator_path and os.path.exists(load_evaluator_path):
            self.model = PPO.load(load_evaluator_path, env=self.env)
            print(f"🔄 RL-evaluator loaded from {load_evaluator_path}")
        else:

            try:
                rollout_steps = self.env.get_attr('evaluator_batch_size')[0]
            except AttributeError:
                rollout_steps = self.env.get_attr('controller_batch_size')[0]

            self.model = PPO(
                policy=CustomGaussianMlpPolicy,
                env=self.env,
                verbose=0,
                n_steps=rollout_steps,
                batch_size=64,
                gamma=1.0,
                learning_rate=1e-4,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                tensorboard_log=tensorboard_log_dir,
            )

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.env.envs[0].policy_model = self.model
        self.env.envs[0].writer = self.writer

    def save_trainset_scores(self, image_ids, save_path):
        """Generates and saves image quality scores deterministically upon policy convergence."""
        features = dict(zip(self.env.envs[0].image_ids, self.env.envs[0].features))
        scores = []
        for img_id in image_ids:
            feat = features[img_id]
            feat = np.expand_dims(feat, axis=0)  # [1, D]
            action, _ = self.model.predict(feat, deterministic=True)
            scores.append(float(action[0][0]))

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        np.savez_compressed(save_path, image_ids=np.array(image_ids), scores=np.array(scores))
        print(f"✅ Train set scores saved to {save_path}")

    def train(self, num_episodes, save_dir="checkpoints"):
        """Trains the RL-evaluator over specified number of episodes."""
        time_steps = int(num_episodes * self.model.n_steps)
        print(f"🚀 Starting RL-evaluator training for {num_episodes} episodes ({time_steps} timesteps)")

        os.makedirs(save_dir, exist_ok=True)

        for episode in trange(num_episodes, desc="Training Episodes", unit="episode", leave=True, ncols=100):
            self.model.learn(total_timesteps=self.model.n_steps, reset_num_timesteps=False)

            moving_avg_window = self.env.get_attr('moving_avg_window')[0]
            reward_window = getattr(self.env.envs[0], 'reward_window', [])

            if moving_avg_window:
                last_weighted_score = moving_avg_window[-1]
                moving_avg = np.mean(moving_avg_window)
            else:
                last_weighted_score = 0.0
                moving_avg = 0.0

            if reward_window:
                last_reward = reward_window[-1]
                reward_avg = np.mean(reward_window)
            else:
                last_reward = 0.0
                reward_avg = 0.0

            self.writer.add_scalar("Metrics/Last_WeightedScore", last_weighted_score, episode)
            self.writer.add_scalar("Metrics/MovingAvg_WeightedScore", moving_avg, episode)
            self.writer.add_scalar("Metrics/Last_Reward", last_reward, episode)
            self.writer.add_scalar("Metrics/Avg_Reward", reward_avg, episode)

            ep_id = episode + 1

            if ep_id % 300 == 0:
                model_save_path = os.path.join(save_dir, f"rl_evaluator_ep{ep_id}.zip")
                self.model.save(model_save_path)
                print(f"\n💾 Saved RL-evaluator checkpoint to {model_save_path}")

            if ep_id % 20 == 0:
                try:
                    feature_npz = self.env.get_attr('feature_npz_path')[0]
                    if os.path.exists(feature_npz):
                        npz_data = np.load(feature_npz)
                        features = npz_data['features']
                        ids = npz_data['image_ids']
                        id_to_index = {k: i for i, k in enumerate(ids)}

                        scores = []
                        for img_id in ids:
                            idx = id_to_index[img_id]
                            feat = features[idx].astype(np.float32)
                            feat = np.expand_dims(feat, axis=0)
                            action, _ = self.model.predict(feat, deterministic=True)
                            scores.append(float(action[0][0]))

                        intermediate_scores_path = os.path.join(save_dir, f"image_scores_ep{ep_id}.npz")
                        np.savez_compressed(intermediate_scores_path, image_ids=np.array(ids), scores=np.array(scores))
                except Exception as e:
                    print(f"\n❌ Failed to save intermediate scores at episode {ep_id}: {e}")

        self.writer.flush()

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        print(f"💾 RL-evaluator saved to {path}")

    def reload(self, load_path):
        self.model = PPO.load(load_path, env=self.env)
        print(f"🔄 RL-evaluator reloaded from {load_path}")