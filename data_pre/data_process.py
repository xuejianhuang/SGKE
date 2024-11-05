import json
import shutil
import  os


def split_json(input_file, split_index,weibo_file, twitter_file):
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确保数据是一个字典
    if isinstance(data, dict):
        # 获取字典的所有项
        items = list(data.items())

        # 分割前10个项和剩余的项
        weibo_items = dict(items[:split_index])
        twitter_items_list = items[split_index:]

        # 将剩余项的键重新编号
        twitter_items = {str(i): value for i, (key, value) in enumerate(twitter_items_list)}

        # 将前10个项写入新的JSON文件
        with open(weibo_file, 'w', encoding='utf-8') as f:
            json.dump(weibo_items, f, ensure_ascii=False, indent=4)

        # 将剩余的项写入另外一个JSON文件
        with open(twitter_file, 'w', encoding='utf-8') as f:
            json.dump(twitter_items, f, ensure_ascii=False, indent=4)
    else:
        print("The data in the JSON file is not a dictionary.")


def split_img(source_dir, split_index,weibo_img_dir, twitter_img_dir):

    # 创建目标目录，如果不存在
    os.makedirs(weibo_img_dir, exist_ok=True)
    os.makedirs(twitter_img_dir, exist_ok=True)

    # 获取源目录中的所有文件名
    files = os.listdir(source_dir)

    # 提取文件名中的数字并排序
    def extract_number(filename):
        return int(filename.split('.')[0])

    # 对文件名进行排序
    files.sort(key=extract_number)

    # 移动文件
    for i, filename in enumerate(files):
        source_path = os.path.join(source_dir, filename)
        if os.path.isfile(source_path):
            if i < split_index:
                destination_path = os.path.join(weibo_img_dir, filename)
            else:
                destination_path = os.path.join(twitter_img_dir, filename)
            shutil.move(source_path, destination_path)


def split_evidence(source_dir, split_index,weibo_evidence_dir, twitter_evidence_dir):

    os.makedirs(weibo_evidence_dir, exist_ok=True)
    os.makedirs(twitter_evidence_dir, exist_ok=True)

    # 获取源目录中的所有子目录名
    directories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # 提取目录名中的数字并排序
    def extract_number(dirname):
        return int(dirname)
    directories.sort(key=extract_number)

    # 初始化计数器
    dir_count = 0
    # 移动目录
    for dirname in directories:
        source_path = os.path.join(source_dir, dirname)

        if dir_count < split_index:
            destination_path = os.path.join(weibo_evidence_dir, dirname)
        else:
            destination_path = os.path.join(twitter_evidence_dir, dirname)

        # 移动目录到目标目录
        shutil.move(source_path, destination_path)
        dir_count += 1

def merge_datasets(train_path, test_path, val_path, merged_path):
    # Load the training data
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # Get the last key from the training data
    if train_data:
        last_key = max(map(int, train_data.keys()))
    else:
        last_key = -1

    # Load the test data and update keys
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_data_updated = {}
    for key, value in test_data.items():
        last_key += 1
        test_data_updated[str(last_key)] = value

    # Load the validation data and update keys
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    val_data_updated = {}
    for key, value in val_data.items():
        last_key += 1
        val_data_updated[str(last_key)] = value

    # Merge all data
    merged_data = {**train_data, **test_data_updated, **val_data_updated}

    # Write the merged data to the output file
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    train_path = '../data/Twitter/dataset_items_train.json'
    test_path = '../data/Twitter/dataset_items_test.json'
    val_path = '../data/Twitter/dataset_items_val.json'
    merged_path = '../data/Twitter/dataset_items_merged.json'

    # Merge the datasets
    merge_datasets(train_path, test_path, val_path, merged_path)