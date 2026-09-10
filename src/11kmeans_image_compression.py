"""使用 MiniBatchKMeans 减少图片中的颜色数量。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.cluster import MiniBatchKMeans


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "materials" / "img1106RowImg.jpg"


def load_image(image_path):
    """读取 RGB 图片，并将像素值归一化至 [0, 1]。"""
    image = io.imread(image_path)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("图片必须至少包含 RGB 三个颜色通道。")

    image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(float) / np.iinfo(image.dtype).max
    else:
        image = np.clip(image.astype(float), 0.0, 1.0)
    return image


def compress_image(pixels, image_shape, color_count):
    """将全部像素聚为指定数量的代表色。"""
    model = MiniBatchKMeans(
        n_clusters=color_count,
        batch_size=max(100, min(1024, pixels.shape[0] // 10)),
        max_iter=20,
        n_init=3,
        random_state=42,
    )
    labels = model.fit_predict(pixels)
    return model.cluster_centers_[labels].reshape(image_shape)


def main():
    image = load_image(DEFAULT_IMAGE_PATH)
    pixels = image.reshape(-1, 3)
    color_counts = [4, 16, 64]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"原图（{len(np.unique(pixels, axis=0))} 色）")
    axes[0].axis("off")

    for axis, color_count in zip(axes[1:], color_counts):
        print(f"正在压缩为 {color_count} 色……")
        compressed = compress_image(pixels, image.shape, color_count)
        axis.imshow(compressed)
        axis.set_title(f"k={color_count}（{color_count} 色）")
        axis.axis("off")

    fig.suptitle("K-Means 图片颜色压缩", fontsize=16)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
