import numpy as np
import torch
from torch import nn
import random

__all__ = [
    'TripletPredictor'
]


class TrainableDistanceEstimator(nn.Module):
    def __init__(self, inplanes, midplanes=256) -> None:
        super().__init__()
        self.distance_func = nn.Sequential(
            nn.Linear(inplanes, midplanes),
            nn.ReLU(True),
            nn.Linear(midplanes, midplanes),
            nn.ReLU(True),
            nn.Linear(midplanes, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        out = self.distance_func(x)

        # we asssume one output node represents dissimilarity between inputs
        out = out[:, [0]]

        return out


class TripletPredictor(nn.Module):
    def __init__(self, n_inputs, n_outputs, task='regression', class_weight=None, include_classifier=True):
        super(TripletPredictor, self).__init__()
        if include_classifier:
            self.classifier_1 = nn.Linear(n_inputs, n_outputs)
            nn.init.xavier_uniform_(self.classifier_1.weight)
            if n_outputs == 2 and class_weight is not None:
                if self.classifier_1.bias is not None:
                    print(
                        "#################### init bias for imb dataset #########################")
                    w0, w1 = class_weight.tolist()
                    nn.init.constant(self.classifier_1.bias, np.log(w1/w0))
        else:
            self.classifier_1 = nn.Identity()

        # discriminates same/differet class
        self.D = nn.Sequential(
            nn.Linear(2*n_inputs, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # trainable distance function
        self.g = TrainableDistanceEstimator(2*n_inputs, 256)
        self.task = task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.class_weight = class_weight
        self.include_classifier = include_classifier
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, *inputs):
        x, targets, enable_tsn, batch_size = inputs
        device = x.device

        x = x.mean((2, 3, 4))  # spatio-temporal average pooling

        out = self.classifier_1(x)
        if enable_tsn:
            # consensus
            out = out.view(
                batch_size, -1, out.size(-1)).mean(1)

        if targets is None:
            return out, 0.0

        # for dummy return vals
        D_loss = torch.tensor(0.0)
        triplet_loss = torch.tensor(0.0)

        if self.training:
            # anchor : random sel, sample until anchor tgt count is bigger than 1.
            while True:
                anc_ix = random.randint(0, x.size(0)-1)
                anc_vec, anc_tgt = x[[anc_ix]], targets[anc_ix]

                if (targets == anc_tgt).sum() > 1:
                    break

            ixs = list(range(x.size(0)))
            ixs.pop(anc_ix)  # exclude anchor

            x_s, targets_s = x[ixs], targets[ixs]
            pos_vec, neg_vec = x_s[targets_s ==
                                   anc_tgt], x_s[targets_s != anc_tgt]

            # match 1st dims
            dim = max(len(pos_vec), len(neg_vec))

            def get_loop_indices(vec):
                loop_indices = list(range(len(vec)))
                for i in loop_indices:
                    if len(loop_indices) >= dim-len(vec):
                        break
                    loop_indices.append(i)

                return loop_indices[:dim-len(vec)]

            try:
                pos_vec = torch.cat(
                    (pos_vec, pos_vec[get_loop_indices(pos_vec)]))
                neg_vec = torch.cat(
                    (neg_vec, neg_vec[get_loop_indices(neg_vec)]))
            except Exception as e:
                import ipdb
                ipdb.set_trace()

            pos_pairs = torch.cat(
                (anc_vec.repeat((pos_vec.size(0), 1)), pos_vec), 1)
            neg_pairs = torch.cat(
                (anc_vec.repeat((neg_vec.size(0), 1)), neg_vec), 1)
            third_pairs = torch.cat((pos_vec, neg_vec), 1)

            # pos(=1.0) : same / neg(=0.0) : not same
            pos_labels = torch.full((pos_pairs.size(0),), 1.0, device=device)
            neg_labels = torch.full((neg_pairs.size(0),), 0.0, device=device)

            bce = nn.functional.binary_cross_entropy

            D_loss = bce(self.D(pos_pairs), pos_labels) + \
                bce(self.D(neg_pairs), neg_labels) +\
                bce(self.D(third_pairs), neg_labels)

            d_anc__pos = self.g(pos_pairs)
            d_anc__neg = self.g(neg_pairs)

            # adaptive margin
            margin = 1.0 * (d_anc__pos.mean()-d_anc__neg.mean())
            triplet_loss = torch.max(torch.tensor(
                0.0, device=device), d_anc__pos-d_anc__neg + margin).mean()

        loss_dict = {
            self.task + "_loss": self.criterion(out, targets),
            'discriminator_loss': D_loss,
            'triplet_loss': triplet_loss
        }

        return out, loss_dict
