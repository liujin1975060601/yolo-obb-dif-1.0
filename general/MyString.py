import os

def replace_path(src_name, last_path, ext):
    # 将源文件名分割为目录、文件名和扩展名
    dirname, filename = os.path.split(src_name)
    base, file_extension = os.path.splitext(filename)

    # 查找并替换最后一个路径部分
    parts = dirname.split(os.path.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = last_path
            break

    # 重新构建目录路径
    new_dirname = os.path.sep.join(parts)

    # 构建新文件名
    new_filename = base + ext

    # 拼接新的文件路径
    dst_name = os.path.join(new_dirname, new_filename)

    return dst_name

def replace_last_path(path,new_folder_name):
    # 将路径拆分成目录和文件名部分
    dirname, basename = os.path.split(path)

    # 将"images"替换为"labels"
    new_path = os.path.join(dirname, new_folder_name)
    return new_path

def add_suffix_to_filename(path, suffix, ext2=None):
    # 分离路径和文件名
    dir_name, base_name = os.path.split(path)
    # 分离文件名和扩展名
    file_name, ext = os.path.splitext(base_name)
    # 添加后缀到文件名
    new_file_name = f"{file_name}{suffix}{ext}" if ext2==None else f"{file_name}{suffix}{ext2}"
    # 重新组合成新的路径
    new_path = os.path.join(dir_name, new_file_name)
    return new_path


def is_valid_file(file_path):
    """
    检查给定路径是否存在，是否是一个文件，并且该文件是否真实存在。

    参数:
    file_path (str): 文件的完整路径。

    返回:
    bool: 如果文件存在并且是一个文件，则返回 True，否则返回 False。
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)