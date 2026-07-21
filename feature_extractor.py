import argparse
import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_feature(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(img).squeeze()
        return feat.cpu().numpy()

    def extract_features_from_folder(self, img_folder, output_path):
        # 自动支持多种常见图像格式（忽略大小写）
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

        all_files = sorted(os.listdir(img_folder))
        img_paths = [
            os.path.join(img_folder, f) for f in all_files
            if f.lower().endswith(valid_extensions)
        ]

        if len(img_paths) == 0:
            print(f"⚠️ Warning: No valid image files found in {img_folder}")
            return

        features = []
        image_ids = []

        for img_path in tqdm(img_paths, desc="Extracting features"):
            feature = self.extract_feature(img_path)
            features.append(feature.flatten())
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            image_ids.append(image_id)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        features = np.stack(features, axis=0)
        np.savez_compressed(output_path, features=features, image_ids=np.array(image_ids))
        print(f"✅ Saved extracted features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ResNet50 features")

    parser.add_argument("--img_folder", type=str, required=True,
                        help="Path to your training images directory (e.g., /your/image/path/).")

    parser.add_argument("--output_path", type=str, default="features.npz",
                        help="Path to save the extracted .npz features (e.g., /your/output/path/features.npz).")

    args = parser.parse_args()

    extractor = FeatureExtractor()
    extractor.extract_features_from_folder(
        img_folder=args.img_folder,
        output_path=args.output_path
    )