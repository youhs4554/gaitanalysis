from .ucf101 import UCF101
import glob
import os


class HMDB51(UCF101):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _select_fold(self, video_list, annotation_path, fold, train):
        target_tag = 1 if train else 2
        name = "*test_split{}.txt".format(fold)
        files = glob.glob(os.path.join(annotation_path, name))
        selected_files = []
        for f in files:
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.strip().split(" ") for x in data]
                data = [x[0] for x in data if int(x[1]) == target_tag]
                selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [
            i
            for i in range(len(video_list))
            if os.path.basename(video_list[i]) in selected_files
        ]
        return indices
