import os
import numpy as np
from PPO_interaction import PPOInterface
from tqdm import trange

# 防止警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __name__ == "__main__":

    base_train_img_dir = r""
    base_train_lbl_dir = r""
    val_img_dir = r""
    val_lbl_dir = r""

    train_img_list = os.listdir(base_train_img_dir)
    image_ids = [os.path.splitext(f)[0] for f in train_img_list if f.endswith('.jpg')]

    save_dir = "temp"
    os.makedirs(save_dir, exist_ok=True)


    controller_load_path = None
    # controller_load_path = os.path.join(save_dir, "ppo_controller_latest")


    controller_save_path = os.path.join(save_dir, "ppo_controller_latest")
    train_scores_save_path = os.path.join(save_dir, "final_image_values.npz")


    interface = PPOInterface(
        image_ids=image_ids,
        base_train_img_dir=base_train_img_dir,
        base_train_lbl_dir=base_train_lbl_dir,
        val_img_dir=val_img_dir,
        val_lbl_dir=val_lbl_dir,
        model_path="yolo11n.pt",
        load_controller_path=controller_load_path
    )


    num_train_episodes = 300
    # print(f"\n🚀 Starting PPO training for {num_train_episodes} episodes...")
    interface.train(num_train_episodes)


    interface.save(controller_save_path)
    print(f"✅ Controller saved to {controller_save_path}")


    interface.save_trainset_scores(image_ids=image_ids, save_path=train_scores_save_path)
    print(f"✅ Train set scores saved to {train_scores_save_path}")
