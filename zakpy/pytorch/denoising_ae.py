import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F


class CategoricalDecoder(nn.Module):
    def __init__(self, category_counts, in_features):
        """

        :param category_counts: like [("column_name", <n_categories>)]
        :param in_features: size of encoded vector being passed to this decoder
        """
        super(CategoricalDecoder, self).__init__()
        self.category_counts = category_counts
        self.out_layers = nn.ModuleDict({
            k: nn.Linear(in_features, v) for k, v in category_counts
        })

    def forward(self, inputs):
        """
        Will pass the batch of encoded vectors to each of the dense layers to get the probability of each category
        for each categorical column (i.e. applies a softmax for each column)

        :param inputs: (N, in_features)
        :return:
        """
        outputs = list()
        for k, _ in self.category_counts:
            proj = self.out_layers[k]
            outputs.append(proj(inputs))

        h = th.cat(outputs, dim=1)
        return h


class CategoricalEncoder(nn.Module):
    def __init__(self, embed_schema, ohe_schema, dropout_noise, n_hidden):
        super(CategoricalEncoder, self).__init__()
        self.embed_schema = embed_schema
        self.ohe_schema = ohe_schema

        self.in_features = sum([e[2] for e in embed_schema]) + sum([o[1] for o in ohe_schema])

        self.embeds = nn.ModuleDict({
            k: nn.Embedding(e_in, e_out)
            for k, e_in, e_out in embed_schema
        })

        self.dropout = nn.Dropout(dropout_noise)
        self.dense = nn.Linear(self.in_features, n_hidden)

    def forward(self, embed_idx, ohes):
        embeds = list()
        for i, (k, _, _) in enumerate(self.embed_schema):
            embeds.append(self.embeds[k](embed_idx[:,i].long()))
        embeds.append(ohes.float())  # the remaining inputs

        h = th.cat(embeds, dim=1)
        #h = self.dropout(h)
        h = self.dense(h)
        h = self.dropout(h)
        return h


class DenoisingAE(nn.Module):
    """
    Note:  When dealing with multiple categoricals, each of which with a different cardinality, it's important to
    understand that cross entropy loss will scale with the number of possible categories.  Specifically, it scales as
    L ~ ln(N), where N is the number of categories.  Therefore, F.cross_entropy for a categorical should be scaled
    by ln(N).  Or for a single row:  F.cross_entropy(preds, y_true).item()/np.log(i)

    """
    def __init__(self, embed_schema, ohe_schema, dropout_noise, n_hidden, scale_by_n=True):

        super(DenoisingAE, self).__init__()
        self._writer = None
        self.scale_by_n = scale_by_n
        self.embed_schema = embed_schema
        self.ohe_schema = ohe_schema

        self._epochs = 0
        self.trainingLosses = list()
        self.testLosses = list()

        self.encoder = CategoricalEncoder(embed_schema, ohe_schema, dropout_noise, n_hidden)

        out_map = [(k, n_cats) for k, n_cats, _ in self.embed_schema]
        out_map += ohe_schema

        self.out_map = out_map

        self.decoder = CategoricalDecoder(
            out_map,
            n_hidden,
        )

    @property
    def writer(self):
        return self._writer

    @writer.setter
    def writer(self, new_writer):
        self._writer = new_writer

    def calc_loss(self, preds, ys, device):
        """
        preds is spread out over the total number of categories:  all the categoricals are concatenated.
        The y is as many columns as there are categorical columns, and the number indicates which category is the
        true one.

        :param preds: (bs, C) where C is sum(N_categoricals*N_cats)
        :param y: (bs, N_cats) where N_cats is number of columns
        :return:
        """
        l = list()

        idx = 0
        for i, (k, n) in enumerate(self.out_map): # e.g. ("city", 1000)
            # Note:  averages the cross-entropy of column i for the whole batch
            loss = F.cross_entropy(preds[:,idx:idx+n].to(device), ys[:,i].to(device), reduction='none')
            if self.scale_by_n:
                loss /= th.log(th.FloatTensor([n]).to(device))
            l.append(loss.reshape((-1,1)))
            idx += n

        return th.cat(l, dim=1)

    def forward(self, embed_idx, ohes):
        h = self.encoder(embed_idx, ohes)
        h = F.relu(h)
        return self.decoder(h)

    def train_loop(self, trainLoader, testLoader, lr, wd, epochs, device):
        n_train = len(trainLoader.dataset)
        n_test = len(testLoader.dataset)

        optimizer = th.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        self.to(device)

        for epoch in range(epochs):

            testLoss = None
            self.eval()
            for i, (embed_idx, ohes, labels) in enumerate(testLoader):
                outputs = self(embed_idx.to(device), ohes.to(device))
                loss = self.calc_loss(outputs, labels, device)

                # print statistics

                if testLoss is not None:
                    testLoss += loss.sum(dim=0).cpu()
                else:
                    testLoss = loss.sum(dim=0).cpu()

                # print(testLoss.shape) # [2]


            trainingLoss = None
            self.train()
            for i, (embed_idx, ohes, labels) in enumerate(trainLoader):

                # forward + backward + optimize
                outputs = self(embed_idx.to(device), ohes.to(device))
                loss = self.calc_loss(outputs, labels, device)

                # zero the parameter gradients
                loss_agg = loss.sum(dim=1).mean()
                optimizer.zero_grad()
                loss_agg.backward()
                optimizer.step()

                if trainingLoss is not None:
                    trainingLoss += loss.sum(dim=0).cpu()
                else:
                    trainingLoss = loss.sum(dim=0).cpu()

            self.trainingLosses.append(trainingLoss.sum().item() / n_train)
            self.testLosses.append(testLoss.sum().item() / n_test)

            self._epochs += 1

            loss_msg = f"Epoch: {self._epochs}, Train Loss: {self.trainingLosses[-1]}, Test Loss: {self.testLosses[-1]}"
            print(loss_msg)

            # Tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/train', self.trainingLosses[-1], self._epochs)
                self.writer.add_scalar('Loss/test', self.testLosses[-1], self._epochs)

                for i, (name, _) in enumerate(self.out_map):
                    self.writer.add_scalar(f'GroupedLoss/test/{name}', testLoss[i] / n_test, self._epochs)

        if self.writer:
            self.writer.flush()

    def extract_embedding(self, dataLoader, device):
        self.eval()
        self.to(device)
        embeds = list()
        for i, (embed_idx, ohes, _) in enumerate(dataLoader):
            outputs = self.encoder(embed_idx.to(device), ohes.to(device)).detach().cpu().numpy()
            embeds.append(outputs)
        return np.concatenate(embeds, axis=0)
