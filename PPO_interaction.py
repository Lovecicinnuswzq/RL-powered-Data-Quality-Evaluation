import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from PPO_env import YoloDVRLEnv
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomGaussianMlpPolicy(ActorCriticPolicy):
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
                 base_train_img_dir, base_train_lbl_dir,
                 val_img_dir, val_lbl_dir,
                 model_path="yolo11n.pt",
                 feature_npz_path="",
                 load_controller_path=None,
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

        if load_controller_path:
            self.model = PPO.load(load_controller_path, env=self.env)
            print(f"🔄 Controller loaded from {load_controller_path}")
        else:
            rollout_steps = self.env.get_attr('controller_batch_size')[0]
            self.model = PPO(
                policy=CustomGaussianMlpPolicy,
                env=self.env,
                verbose=0,
                n_steps=rollout_steps,
                batch_size=20,
                gamma=1.0,
                learning_rate=1e-4,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                tensorboard_log=tensorboard_log_dir,
            )

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.env.envs[0].policy_model = self.model
        self.env.envs[0].writer = self.writer

    def save_trainset_scores(self, image_ids, save_path):
        features = dict(zip(self.env.envs[0].image_ids, self.env.envs[0].features))
        scores = []
        for img_id in image_ids:
            feat = features[img_id]
            feat = np.expand_dims(feat, axis=0)  # [1, D]
            action, _ = self.model.predict(feat, deterministic=True)
            scores.append(float(action[0][0]))

        np.savez_compressed(save_path, image_ids=np.array(image_ids), scores=np.array(scores))
        print(f"✅ Train set scores saved to {save_path}")

    def train(self, num_episodes):
        time_steps = int(num_episodes * self.model.n_steps)
        print(f"🚀 Starting training for {num_episodes} episodes ({time_steps} timesteps)")

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

            print(
                f"🌟 Episode {episode + 1}/{num_episodes} | Last Weighted Score (AP Proxy): {last_weighted_score:.4f} | Moving Avg: {moving_avg:.4f} | Last Reward: {last_reward:.4f} | Reward Avg: {reward_avg:.4f}")
            self.writer.add_scalar("Metrics/Last_WeightedScore", last_weighted_score, episode)
            self.writer.add_scalar("Metrics/MovingAvg_WeightedScore", moving_avg, episode)
            self.writer.add_scalar("Metrics/Last_Reward", last_reward, episode)
            self.writer.add_scalar("Metrics/Avg_Reward", reward_avg, episode)

            ep_id = episode + 1
            if ep_id % 300 == 0:
                save_dir = "temp"
                os.makedirs(save_dir, exist_ok=True)
                model_save_path = os.path.join(save_dir, f"dvrl_model_ep{ep_id}.zip")
                self.model.save(model_save_path)
                print(f"💾 Saved model to {model_save_path}")

            if ep_id % 20 == 0:
                save_dir = "temp"
                os.makedirs(save_dir, exist_ok=True)

                try:
                    npz_data = np.load(self.env.get_attr('feature_npz_path')[0])
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

                    save_path = os.path.join(save_dir, f"image_scores_ep{ep_id}.npz")
                    np.savez_compressed(save_path, image_ids=np.array(ids), scores=np.array(scores))
                    print(f"✅ Train set scores saved to {save_path}")
                except Exception as e:
                    print(f"❌ Failed to save at episode {ep_id}: {e}")

        self.writer.flush()

    def save(self, path):
        self.model.save(path)
        print(f"💾 Controller saved to {path}")

    def reload(self, load_path):
        self.model = PPO.load(load_path, env=self.env)
        print(f"🔄 Controller reloaded from {load_path}")
