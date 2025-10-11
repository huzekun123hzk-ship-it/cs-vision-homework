# knn/experiments/image_folder_utils.py

"""
用于处理简单图像文件夹数据集的辅助工具。

包含一个最小化的下载器和解压器，可以从一个指向 ZIP 或 TAR 压缩包的 URL
下载文件，并将其解压到目标目录。
预期的压缩包内容：包含图像的类别子文件夹。
"""
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# --- 私有辅助函数，用于判断压缩包类型 ---

def _is_zip(path: Path) -> bool:
    """判断文件是否为 .zip 格式。"""
    return path.suffix.lower() == ".zip"

def _is_targz(path: Path) -> bool:
    """判断文件是否为 .tar.gz, .tar.bz2, .tar.xz 等格式。"""
    s = path.suffixes
    return len(s) >= 2 and s[-2].lower() == ".tar" and s[-1].lower() in {".gz", ".bz2", ".xz"}

def _is_tgz(path: Path) -> bool:
    """判断文件是否为 .tgz 格式。"""
    return path.suffix.lower() == ".tgz"

def _is_tar(path: Path) -> bool:
    """判断文件是否为 .tar 格式。"""
    return path.suffix.lower() == ".tar"

# --- 公开函数 ---

def download_and_extract(url: str, dest_dir: str) -> None:
    """
    从指定的 URL 下载一个压缩包并解压到目标目录。

    支持 .zip, .tar, .tar.gz, .tgz 等格式。如果目标目录不存在，将会被创建。
    如果目标目录中已经包含文件或子目录，为了避免重复工作，下载步骤将被跳过。
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # 如果目标目录已有内容，则跳过下载和解压
    try:
        has_content = any(dest.iterdir())
    except FileNotFoundError:
        has_content = False
    if has_content:
        print(f"✓ 目标目录 {dest.resolve()} 非空，跳过下载和解压。")
        return

    # 使用临时目录进行下载，确保过程干净
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "dataset_archive"
        print(f"正在下载 {url} -> {tmp_path}")
        urllib.request.urlretrieve(url, tmp_path)

        # 判断压缩包类型并解压
        print(f"正在解压文件...")
        if _is_zip(tmp_path):
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(dest)
        elif _is_tgz(tmp_path) or _is_targz(tmp_path) or _is_tar(tmp_path):
            with tarfile.open(tmp_path, "r:*") as tf:
                tf.extractall(dest)
        else:
            # 如果后缀不明确，尝试作为 zip 打开，失败则后备为 tar
            try:
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.extractall(dest)
            except zipfile.BadZipFile:
                try:
                    with tarfile.open(tmp_path, "r:*") as tf:
                        tf.extractall(dest)
                except tarfile.TarError as e:
                    raise RuntimeError(f"不支持或已损坏的压缩包格式: {url}") from e

        print(f"✓ 数据集已成功解压至 {dest.resolve()}")