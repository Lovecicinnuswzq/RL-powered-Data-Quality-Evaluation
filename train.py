import argparse
import os
import numpy as np
from PPO_interaction import PPOInterface
from tqdm import trange


def main():
    parser = argparse.ArgumentParser(
        description="Train the RL-evaluator for data quality evaluation."
    )

    # 路径相关参数 (Path Arguments)
    parser.add_argument(
        "--train_img_dir",
        type=str,
        required=True,
        help="Path to the training images directory.",
    )
    parser.add_argument(
        "--train_lbl_dir",
        type=str,
        required=True,
        help="Path to the training labels directory.",
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        required=True,
        help="Path to the validation images directory.",
    )
    parser.add_argument(
        "--val_lbl_dir",
        type=str,
        required=True,
        help="Path to the validation labels directory.",
    )

    # 模型与输出相关参数 (Model & Output Arguments)
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo11n.pt",
        help="Path or name of the initial detector checkpoint (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save the trained RL-evaluator and output scores (default: checkpoints).",
    )
    parser.add_argument(
        "--load_evaluator_path",
        type=str,
        default=None,
        help="Path to resume training from an existing RL-evaluator checkpoint (optional).",
    )

    # 训练超参数 (Hyperparameters)
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of PPO training episodes (default: 1000).",
    )

    args = parser.parse_args()

    # 检查输入的路径是否存在
    for path, name in [
        (args.train_img_dir, "train_img_dir"),
        (args.train_lbl_dir, "train_lbl_dir"),
        (args.val_img_dir, "val_img_dir"),
        (args.val_lbl_dir, "val_lbl_dir"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ Error: The specified {name} does not exist: {path}"
            )

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    )
    train_img_list = sorted(
        [
            f
            for f in os.listdir(args.train_img_dir)
            if f.lower().endswith(valid_extensions)
        ]
    )

    if not train_img_list:
        raise ValueError(
            f"❌ Error: No valid image files found in {args.train_img_dir}"
        )

    image_ids = [os.path.splitext(f)[0] for f in train_img_list]
    print(f"📦 Successfully indexed {len(image_ids)} training images.")

    os.makedirs(args.save_dir, exist_ok=True)
    evaluator_save_path = os.path.join(args.save_dir, "rl_evaluator_latest")
    train_scores_save_path = os.path.join(
        args.save_dir, "final_image_values.npz"
    )

    print("🛠️ Initializing PPO Interface...")
    interface = PPOInterface(
        image_ids=image_ids,
        base_train_img_dir=args.train_img_dir,
        base_train_lbl_dir=args.train_lbl_dir,
        val_img_dir=args.val_img_dir,
        val_lbl_dir=args.val_lbl_dir,
        model_path=args.model_path,
        load_controller_path=args.load_evaluator_path,
    )

    print(f"\n🚀 Starting PPO training for {args.num_episodes} episodes...")
    interface.train(args.num_episodes)

    interface.save(evaluator_save_path)
    print(f"✅ Trained RL-evaluator saved to: {evaluator_save_path}")

    print("📊 Generating final data quality scores for data curation...")
    interface.save_trainset_scores(
        image_ids=image_ids, save_path=train_scores_save_path
    )
    print(f"✅ Final image quality scores saved to: {train_scores_save_path}")


if __name__ == "__main__":
    main()