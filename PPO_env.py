import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import shutil
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import torch
import subprocess
import json


class YoloDVRLEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 image_ids,
                 base_train_img_dir, base_train_lbl_dir,
                 val_img_dir, val_lbl_dir,
                 feature_npz_path="",
                 model_path="yolo11n.pt",
                 temp_dir="temp",
                 controller_batch_size=100,
                 min_selected=0,
                 moving_avg_len=5,
                 log_dir="runs_dvrl"):
        super().__init__()

        self.episode_idx = 0
        self.policy_model = None
        self.writer = SummaryWriter(log_dir=log_dir)

        self.feature_npz_path = feature_npz_path
        self.val_feature_path = ""

        self.image_ids = np.array(image_ids)
        self.train_img_dir = base_train_img_dir
        self.train_lbl_dir = base_train_lbl_dir
        self.val_img_dir = val_img_dir
        self.val_lbl_dir = val_lbl_dir
        self.model_path = model_path
        self.min_selected = min_selected

        self.ema_score = 0.0
        self.use_ema = True
        self.ema_alpha = 0.1

        self.temp_img_dir = os.path.join(temp_dir, "train", "images")
        self.temp_lbl_dir = os.path.join(temp_dir, "train", "labels")
        self.yaml_path = os.path.join(temp_dir, "train.yaml")

        os.makedirs(self.temp_img_dir, exist_ok=True)
        os.makedirs(self.temp_lbl_dir, exist_ok=True)

        with open(self.yaml_path, "w") as f:
            f.write(f"train: {os.path.abspath(self.temp_img_dir)}\n")
            f.write(f"val: {os.path.abspath(self.val_img_dir)}\n")
            f.write("nc: 1\n")
            f.write("names: ['worker']\n")

        self.controller_batch_size = controller_batch_size
        self.moving_avg_len = moving_avg_len
        self.actions_list = []
        self.moving_avg_len = moving_avg_len
        self.moving_avg_window = []
        self.sample_num_count = 0
        self.current_batch = None

        feat_data = np.load(feature_npz_path)
        self.feature_dict = {k: v for k, v in zip(feat_data['image_ids'], feat_data['features'])}
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feat_data['features'].shape[1],),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def get_batch(self):
        rng = np.random.default_rng()
        shuffle_inds = rng.permutation(len(self.image_ids))
        return self.image_ids[shuffle_inds[:self.controller_batch_size]]

    def compute_moving_avg(self):
        if len(self.moving_avg_window) == 0:
            return 0
        return np.mean(self.moving_avg_window[-self.moving_avg_len:])

    def select_samples(self, actions_list, batch_ids):
        actions_list = np.clip(actions_list, 0, 1)
        selection_vector = np.random.binomial(1, actions_list)
        selected_ids = batch_ids[selection_vector.astype(bool)]
        if len(selected_ids) < self.min_selected:
            selected_ids = batch_ids
        return selected_ids

    def _copy_selected_images(self, selected_ids):
        for f in os.listdir(self.temp_img_dir):
            os.remove(os.path.join(self.temp_img_dir, f))
        for f in os.listdir(self.temp_lbl_dir):
            os.remove(os.path.join(self.temp_lbl_dir, f))
        for img_id in selected_ids:
            shutil.copy(os.path.join(self.train_img_dir, f"{img_id}.jpg"),
                        os.path.join(self.temp_img_dir, f"{img_id}.jpg"))
            shutil.copy(os.path.join(self.train_lbl_dir, f"{img_id}.txt"),
                        os.path.join(self.temp_lbl_dir, f"{img_id}.txt"))

    def _compute_iou_xywh(self, box1, box2):
        """
        box1, box2: [cx, cy, w, h] in same coordinate system (pixels)
        """
        try:
            box1 = np.array(box1, dtype=np.float32)
            box2 = np.array(box2, dtype=np.float32)


            box1[2] = max(0, box1[2])
            box1[3] = max(0, box1[3])
            box2[2] = max(0, box2[2])
            box2[3] = max(0, box2[3])

            b1_x1 = box1[0] - box1[2] / 2
            b1_y1 = box1[1] - box1[3] / 2
            b1_x2 = box1[0] + box1[2] / 2
            b1_y2 = box1[1] + box1[3] / 2

            b2_x1 = box2[0] - box2[2] / 2
            b2_y1 = box2[1] - box2[3] / 2
            b2_x2 = box2[0] + box2[2] / 2
            b2_y2 = box2[1] + box2[3] / 2

            inter_x1 = max(b1_x1, b2_x1)
            inter_y1 = max(b1_y1, b2_y1)
            inter_x2 = min(b1_x2, b2_x2)
            inter_y2 = min(b1_y2, b2_y2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
            union_area = b1_area + b2_area - inter_area

            return inter_area / (union_area + 1e-8)  # 避免除以0
        except Exception as e:
            print(f"Error computing IoU: {e}")
            return 0.0

    def _train_predictor_and_get_reward(self, selected_ids):
        self._copy_selected_images(selected_ids)


        cmd = [
            "python", "yolo_trainer.py",
            "--model_path", self.model_path,
            "--yaml_path", self.yaml_path,
            "--val_dir", self.val_img_dir,
            "--output_json", "temp/yolo_val_preds.json"
        ]
        subprocess.run(cmd, check=True)


        with open("temp/yolo_val_preds.json", "r") as f:
            val_preds = json.load(f)


        with open("temp/yolo_val_metrics.json", "r") as f:
            val_metrics = json.load(f)
        map50 = val_metrics.get("map50", 0.0)
        map5095 = val_metrics.get("map5095", 0.0)
        map75 = val_metrics.get("map75", 0.0)

        weighted_score = map50


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
            if self.episode_idx == 0:
                reward = 0.0
            else:
                reward = (weighted_score - moving_avg)

        print(f"[Episode {self.episode_idx}] Performance: {weighted_score:.4f}")
        print(f"[Episode {self.episode_idx}] Moving Avg: {moving_avg:.4f}")
        print(f"[Episode {self.episode_idx}] Reward: {reward:.4f}")


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
        import gc
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
                    print(f"❌ Failed to remove {item_path} due to {e}")

    def step(self, action):
        self.actions_list.append(float(action))
        self.sample_num_count += 1

        if self.sample_num_count < self.controller_batch_size:
            obs = self.feature_dict[self.current_batch[self.sample_num_count]].astype(np.float32)
            return obs, 0, False, False, {}
        else:
            actions_array = np.array(self.actions_list)
            print(f"\n📊 PPO Action Score Summary before Sampling:")
            print(f"Min: {actions_array.min():.4f}, Max: {actions_array.max():.4f}, Mean: {actions_array.mean():.4f}")
            hist, _ = np.histogram(actions_array, bins=20, range=(0, 1))
            print("Histogram:", hist.tolist())
            if hasattr(self, "writer") and self.writer is not None:
                import torch
                self.writer.add_histogram("Controller/ActionDistribution", torch.tensor(actions_array),
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

