import os
from PIL import Image


def check_png_files(directory):
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(directory):
        print(f'{len(files)} files.')
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    # 尝试打开并加载PNG文件
                    with Image.open(file_path) as img:
                        img.verify()  # 仅验证文件，而不实际加载到内存中
                except (IOError, SyntaxError) as e:
                    print(f"文件损坏: {file_path} - 错误信息: {e}")


# 使用示例
check_png_files('/data/yuzhi/Spatial-ViT-main/datasets/voc/masks')  # 将 'path_to_your_directory' 替换为你的PNG文件所在目录
