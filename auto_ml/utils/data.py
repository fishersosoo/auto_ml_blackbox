# coding=utf-8
import pandas as pd
import os
import json


class TNewsData:
    """
    TNEWS新闻标题分类数据集
    [label_id, label, label_desc, sentence, keywords]
    """

    def __init__(self, dir=None, source_dir=None):
        self.num_class = 15
        if dir is not None:
            self.training_set = pd.read_csv(os.path.join(dir, "train.csv"), index_col=None, encoding='utf-8')
            self.test_set = pd.read_csv(os.path.join(dir, "test.csv"), index_col=False, encoding='utf-8')
            self.eval_set = pd.read_csv(os.path.join(dir, "eval.csv"), index_col=False, encoding='utf-8')
        elif source_dir is not None:
            self.read_source_dir(source_dir)
        else:
            raise Exception("dir or source_dir needed")

    @staticmethod
    def _from_json(path):
        data = dict()
        with open(path, encoding='utf-8')as fd:
            text = fd.read()
            lines = text.split('\n')
            for line in lines:
                if len(line) < 2:
                    continue
                for k, v in json.loads(line).items():
                    if k not in data:
                        data[k] = []
                    data[k].append(v)
        df = pd.DataFrame(data)
        return df

    def __getitem__(self, item):
        if item == "train":
            return self.training_set
        if item == "test":
            return self.test_set
        if item == "eval":
            return self.eval_set
        raise KeyError(f"key {item} not found in {type(self)}")

    def read_source_dir(self, dir):
        dev_set = TNewsData._from_json(os.path.join(dir, "dev.json"))
        test_set = TNewsData._from_json(os.path.join(dir, "test.json"))
        training_set = TNewsData._from_json(os.path.join(dir, "train.json"))
        labels = TNewsData._from_json(os.path.join(dir, "labels.json"))
        labels["label_id"] = labels.index
        self.num_class = max(labels.index)
        labels = labels.drop(columns=['label_desc'])
        labels = labels.set_index('label')
        dev_set = dev_set.join(labels, on="label")
        training_set = training_set.join(labels, on="label")
        self.training_set = training_set
        self.test_set = test_set
        self.eval_set = dev_set

    def export_dataset(self, output_dir):
        self.training_set.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding='utf-8')
        self.test_set.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding='utf-8')
        self.eval_set.to_csv(os.path.join(output_dir, "eval.csv"), index=False, encoding='utf-8')


if __name__ == '__main__':
    data = TNewsData(source_dir="/home/auto_ml/data/source/chineseGLUEdatasets.v0.0.1/tnews_public")
    data.export_dataset('/home/auto_ml/data/source/TNewsData')
