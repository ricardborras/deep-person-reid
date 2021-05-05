from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import glob
import numpy as np
import random

from torchreid.data.datasets import ImageDataset
from torchreid.utils import read_json, write_json

# Min images for a subject to consider it
MIN_IMAGES_PER_SUBJECT = 3

class BusReid(ImageDataset):
    """Counterest bus reid.

    This dataset contains a folder named images. Inside images a folder with images from the same subject
    2020-12-16-13.03.47_cam2.ts_7_1014.jpg

    camX indicates camera number (1 or 2)

    For each subject, number of images must be balanced

    """
    dataset_dir = 'bus_reid'
    dataset_url = ''

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)

        self.subjects_dir = osp.join(self.dataset_dir, 'images')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        
        required_files = [
            self.dataset_dir,
            self.subjects_dir
        ]
        self.check_before_run(required_files)
        
        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError('split_id exceeds range, received {}, '
                             'but expected between 0 and {}'.format(split_id, len(splits)-1))
        split = splits[split_id]

        train = split['train']
        query = split['query'] # query and gallery share the same images
        gallery = split['gallery']

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        super(BusReid, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating 10 random splits of train ids and test ids')

            # validate that we have at least 3 images for a subject to consider it valid
            subject_ids = []
            num_images_per_subject = []
            for subject in os.listdir(self.subjects_dir):
                num_subject_images = len(os.listdir(os.path.join(self.subjects_dir, subject)))
                if num_subject_images < MIN_IMAGES_PER_SUBJECT:
                    continue
                subject_ids.append(subject)
                num_images_per_subject.append(num_subject_images)

            # to avoid to have too much images for a given subject, keep a max of the
            max_images_per_subject = int(np.percentile(np.array(num_images_per_subject), 20))

            num_pids = len(subject_ids)
            print('Number of identities: {}, max images per identity {}'.format(num_pids, max_images_per_subject))
            num_train_pids = num_pids // 2

            splits = []
            for _ in range(10):
                np.random.shuffle(subject_ids)
                train_pids = subject_ids[:num_train_pids]
                test_pids = subject_ids[num_train_pids:]
                assert not bool(set(train_pids) & set(test_pids)), 'Error: train and test overlap'

                train = []
                for pid, subject_id in enumerate(train_pids):
                    subject_images = sorted(glob.glob(os.path.join(self.subjects_dir, subject_id, '*.jpg')))
                    # filter to max number of images
                    if len(subject_images) > max_images_per_subject:
                        subject_images = random.sample(subject_images, max_images_per_subject)
                    cam_ids = [BusReid.extract_cam_from_filename(f) for f in subject_images]
                    for img, cam_id in zip(subject_images, cam_ids):
                        if not cam_id:
                            continue
                        train.append((img, pid, cam_id))

                test_gallery = []
                test_query = []
                for pid, subject_id in enumerate(test_pids):
                    subject_images = sorted(glob.glob(os.path.join(self.subjects_dir, subject_id, '*.jpg')))
                    # filter to max number of images
                    if len(subject_images) > max_images_per_subject:
                        subject_images = random.sample(subject_images, max_images_per_subject)
                    cam_ids = [BusReid.extract_cam_from_filename(f) for f in subject_images]
                    for idx, (img, cam_id) in enumerate(zip(subject_images, cam_ids)):
                        if not cam_id:
                            continue
                        # use half of images for query and half for gallery. Query and gallery must come from different
                        # cam ids so, in theory, we should split images per camera, not by index
                        print('Warning!!! Query and gallery are forced to have diferent cameras!')
                        if idx % 2 == 0:
                            test_query.append((img, pid, 0))
                        else:
                            test_gallery.append((img, pid, 1))

                split = {
                    'train': train,
                    'query': test_query,
                    'gallery': test_gallery,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

                # make another split with gallery and query exchanged

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))

    @classmethod
    def extract_cam_from_filename(cls, filename):
        c = filename.find('cam')
        if c < 0:
            return None
        try:
            return int(filename[c+3])
        except ValueError:
            return None
