import os
import json
import shutil
import subprocess
import gc
import cv2
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter


class YoloDVRLEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 image_ids,
                 base_train_img_dir,
                 base_train_lbl_dir,
                 val_img_dir,
                 val_lbl_dir,
                 feature_npz_path="features.npz",
                 model_path="yolo11n.pt",
                 temp_dir="temp",
                 evaluator_batch_size=100,
                 min_selected=0,
                 moving_avg_len=5,
                 log_dir="runs_dvrl"):
        super().__init__()

        self.episode_idx = 0
        self.policy_model = None
        self.writer = SummaryWriter(log_dir=log_dir)

        self.feature_npz_path = feature_npz_path
        self.image_ids = np.array(image_ids)
        self.train_img_dir = base_train_img_dir
        self.train_lbl_dir = base_train_lbl_dir
        self.val_img_dir = val_img_dir
        self.val_lbl_dir = val_lbl_dir
        self.model_path = model_path
        self.min_selected = min_selected
        self.temp_dir = temp_dir

        self.ema_score = 0.0
        self.use_ema = True
        self.ema_alpha = 0.1

        # 挂载临时数据训练路径
        self.temp_img_dir = os.path.join(temp_dir, "train", "images")
        self.temp_lbl_dir = os.path.join(temp_dir, "train", "labels")
        self.yaml_path = os.path.join(temp_dir, "train.yaml")

        os.makedirs(self.temp_img_dir, exist_ok=True)
        os.makedirs(self.temp_lbl_dir, exist_ok=True)

        # 动态创建 YOLO 训练需要的 yaml 配置
        with open(self.yaml_path, "w") as f:
            f.write(f"train: {os.path.abspath(self.temp_img_dir)}\n")
            f.write(f"val: {os.path.abspath(self.val_img_dir)}\n")
            f.write("nc: 80\n")  # 默认设置为通用数据集配置 (用户可自定义)
            f.write("names: ['object']\n")

        self.evaluator_batch_size = evaluator_batch_size
        self.controller_batch_size = evaluator_batch_size  # 兼容旧属性访问

        self.moving_avg_len = moving_avg_len
        self.actions_list = []
        self.moving_avg_window = []
        self.sample_num_count = 0
        self.current_batch = None

        # 读取提取好的特征文件
        if not os.path.exists(feature_npz_path):
            raise FileNotFoundError(f"❌ Feature file not found at: {feature_npz_path}")

        feat_data = np.load(feature_npz_path)
        self.feature_dict = {k: v for k, v in zip(feat_data['image_ids'], feat_data['features'])}

        # 动作空间与状态空间定义
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feat_data['features'].shape[1],), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def get_batch(self):
        rng = np.random.default_rng()
        shuffle_inds = rng.permutation(len(self.image_ids))
        return self.image_ids[shuffle_inds[:self.evaluator_batch_size]]

    def compute_moving_avg(self):
        if len(self.moving_avg_window) == 0:
            return 0.0
        return float(np.mean(self.moving_avg_window[-self.moving_avg_len:]))

    def select_samples(self, actions_list, batch_ids):
        actions_list = np.clip(actions_list, 0, 1)
        selection_vector = np.random.binomial(1, actions_list)
        selected_ids = batch_ids[selection_vector.astype(bool)]
        if len(selected_ids) < self.min_selected:
            selected_ids = batch_ids
        return selected_ids

    def _copy_selected_images(self, selected_ids):
        # 清空临时文件夹
        for f in os.listdir(self.temp_img_dir):
            os.remove(os.path.join(self.temp_img_dir, f))
        for f in os.listdir(self.temp_lbl_dir):
            os.remove(os.path.join(self.temp_lbl_dir, f))

        # 查找图片格式并拷贝
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for img_id in selected_ids:
            # 匹配对应后缀的图像文件
            img_name = None
            for ext in valid_exts:
                if os.path.exists(os.path.join(self.train_img_dir, f"{img_id}{ext}")):
                    img_name = f"{img_id}{ext}"
                    break

            if img_name:
                shutil.copy(os.path.join(self.train_img_dir, img_name),
                            os.path.join(self.temp_img_dir, img_name))

            lbl_name = f"{img_id}.txt"
            lbl_path = os.path.join(self.train_lbl_dir, lbl_name)
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(self.temp_lbl_dir, lbl_name))

    def _train_predictor_and_get_reward(self, selected_ids):
        self._copy_selected_images(selected_ids)

        preds_json_path = os.path.join(self.temp_dir, "yolo_val_preds.json")
        metrics_json_path = os.path.join(self.temp_dir, "yolo_val_metrics.json")

        cmd = [
            "python", "train_yolo_subprocess.py",
            "--model_path", self.model_path,
            "--yaml_path", self.yaml_path,
            "--val_dir", self.val_img_dir,
            "--output_json", preds_json_path
        ]
        subprocess.run(cmd, check=True)

        # 读取评估指标
        map50 = 0.0
        map5095 = 0.0
        map75 = 0.0

        if os.path.exists(metrics_json_path):
            with open(metrics_json_path, "r") as f:
                val_metrics = json.load(f)
            map50 = val_metrics.get("map50", 0.0)
            map5095 = val_metrics.get("map5095", 0.0)
            map75 = val_metrics.get("map75", 0.0)

        weighted_score = map50

        # 计算 Moving Average 与 Reward
        if self.use_ema:
            if self.episode_idx < self.moving_avg_len:
                self.moving_avg_window.append(weighted_score)
                moving_avg = np.mean(self.moving_avg_window)
                reward = (weighted_score - moving_avg)
                self.ema_score = moving_avg
            else:
                reward = (weighted_score - self.ema_score)
                self.ema_score = self.ema_alpha * weighted_score + (1 - self.ema_alpha) * self.ema_score
                moving_avg = self.ema_score
        else:
            self.moving_avg_window.append(weighted_score)
            if len(self.moving_avg_window) > self.moving_avg_len:
                self.moving_avg_window.pop(0)

            moving_avg = np.mean(self.moving_avg_window)
            reward = 0.0 if self.episode_idx == 0 else (weighted_score - moving_avg)

        print(
            f"[Episode {self.episode_idx}] Performance (mAP50): {weighted_score:.4f} | Moving Avg: {moving_avg:.4f} | Reward: {reward:.4f}")

        # 写入 TensorBoard 记录
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_scalar("val/map50", map50, self.episode_idx)
            self.writer.add_scalar("val/map5095", map5095, self.episode_idx)
            self.writer.add_scalar("val/map75", map75, self.episode_idx)
            self.writer.add_scalar("Metrics/Performance", weighted_score, self.episode_idx)
            self.writer.add_scalar("Metrics/Moving_Avg", moving_avg, self.episode_idx)
            self.writer.add_scalar("Metrics/Reward", reward, self.episode_idx)

            if not hasattr(self, "reward_window"):
                self.reward_window = []
            self.reward_window.append(reward)
            avg_reward = np.mean(self.reward_window)
            self.writer.add_scalar("Metrics/Avg_Reward", avg_reward, self.episode_idx)

        self._cleanup_yolo_runs()
        gc.collect()
        torch.cuda.empty_cache()
        return reward, weighted_score, moving_avg

    def _cleanup_yolo_runs(self):
        runs_dir = os.path.join(os.getcwd(), 'runs')
        if os.path.exists(runs_dir):
            for item in os.listdir(runs_dir):
                item_path = os.path.join(runs_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    pass

    def step(self, action):
        self.actions_list.append(float(action))
        self.sample_num_count += 1

        if self.sample_num_count < self.evaluator_batch_size:
            obs = self.feature_dict[self.current_batch[self.sample_num_count]].astype(np.float32)
            return obs, 0.0, False, False, {}
        else:
            actions_array = np.array(self.actions_list)
            print(
                f"\n📊 Action Score Distribution (RL-evaluator): Min={actions_array.min():.4f}, Max={actions_array.max():.4f}, Mean={actions_array.mean():.4f}")

            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_histogram("RL_Evaluator/ActionDistribution", torch.tensor(actions_array),
                                          self.episode_idx)
                if self.episode_idx % 10 == 0:
                    self.writer.flush()

            selected_ids = self.select_samples(actions_array, self.current_batch)
            reward, map50, moving_avg = self._train_predictor_and_get_reward(selected_ids)

            self.episode_idx += 1
            obs = self.feature_dict[self.current_batch[0]].astype(np.float32)
            info = {"mAP50": map50, "moving_avg": moving_avg, "num_selected": len(selected_ids)}
            self.latest_info = info
            return obs, reward, True, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_batch = self.get_batch()
        self.actions_list = []
        self.sample_num_count = 0
        obs = self.feature_dict[self.current_batch[self.sample_num_count]].astype(np.float32)
        return obs, {}