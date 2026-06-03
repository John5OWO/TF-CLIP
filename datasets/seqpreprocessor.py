from __future__ import absolute_import
import os.path as osp
import torch
from PIL import Image


class SeqTrainPreprocessor(object):
    def __init__(self, seqset, dataset, seq_len, transform=None):
        super(SeqTrainPreprocessor, self).__init__()
        self.seqset = seqset
        self.identities = dataset.identities
        self.transform = transform
        self.seq_len = seq_len
        self.root = [dataset.images_dir]
        self.has_flow = osp.isdir(dataset.other_dir)
        if self.has_flow:
            self.root.append(dataset.other_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        start_ind, end_ind, pid, label, camid = self.seqset[index]

        imgseq = []
        flowseq = []
        for ind in range(start_ind, end_ind):
            fname = self.identities[pid][camid][ind]
            fpath_img = osp.join(self.root[0], fname)
            imgrgb = Image.open(fpath_img).convert('RGB')
            imgseq.append(imgrgb)
            if self.has_flow:
                fpath_flow = osp.join(self.root[1], fname)
                flowrgb = Image.open(fpath_flow).convert('RGB')
                flowseq.append(flowrgb)

        while len(imgseq) < self.seq_len:
            imgseq.append(imgrgb)
            if self.has_flow:
                flowseq.append(flowrgb)

        if self.has_flow:
            seq = self.transform([imgseq, flowseq]) if self.transform else [imgseq, flowseq]
            img_tensor = torch.stack(seq[0], 0)
            flow_tensor = torch.stack(seq[1], 0)
        else:
            seq = self.transform([imgseq]) if self.transform else [imgseq]
            img_tensor = torch.stack(seq[0], 0)
            flow_tensor = None

        return img_tensor, flow_tensor, label, camid


class SeqTestPreprocessor(object):

    def __init__(self, seqset, dataset, seq_len, transform=None):
        super(SeqTestPreprocessor, self).__init__()
        self.seqset = seqset
        self.identities = dataset.identities
        self.transform = transform
        self.seq_len = seq_len
        self.root = [dataset.images_dir]
        self.has_flow = osp.isdir(dataset.other_dir)
        if self.has_flow:
            self.root.append(dataset.other_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        start_ind, end_ind, pid, label, camid = self.seqset[index]

        imgseq = []
        flowseq = []
        for ind in range(start_ind, end_ind):
            fname = self.identities[pid][camid][ind]
            fpath_img = osp.join(self.root[0], fname)
            imgrgb = Image.open(fpath_img).convert('RGB')
            imgseq.append(imgrgb)
            if self.has_flow:
                fpath_flow = osp.join(self.root[1], fname)
                flowrgb = Image.open(fpath_flow).convert('RGB')
                flowseq.append(flowrgb)

        while len(imgseq) < self.seq_len:
            imgseq.append(imgrgb)
            if self.has_flow:
                flowseq.append(flowrgb)

        if self.has_flow:
            seq = self.transform([imgseq, flowseq]) if self.transform else [imgseq, flowseq]
            img_tensor = torch.stack(seq[0], 0)
            flow_tensor = torch.stack(seq[1], 0)
        else:
            seq = self.transform([imgseq]) if self.transform else [imgseq]
            img_tensor = torch.stack(seq[0], 0)
            flow_tensor = None

        return img_tensor, flow_tensor, pid, camid
