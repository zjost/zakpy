import torch as th
from torch import nn
from torch.nn import functional as F


class CategoricalDecoder(nn.Module):
    def __init__(self, category_count_map, in_features):
        """

        :param category_count_map: like {"column_name": <n_categories>}
        :param in_features: size of encoded vector being passed to this decoder
        """
        super(CategoricalDecoder, self).__init__()
        self.out_layers = {
            k: nn.Linear(in_features, v) for k, v in category_count_map.items()
        }

    def forward(self, inputs):
        """
        Will pass the batch of encoded vectors to each of the dense layers to get the probability of each category
        for each categorical column (i.e. applies a softmax for each column)

        :param inputs: (N, in_features)
        :return:
        """
        outputs = list()
        for k, proj in self.out_layers.items():
            outputs.append(F.softmax(proj(inputs), dim=1))

        h = th.cat(outputs, dim=1)
        return h


class DenoisingAE(nn.Module):
    """
    Note:  When dealing with multiple categoricals, each of which with a different cardinality, it's important to
    understand that cross entropy loss will scale with the number of possible categories.  Specifically, it scales as
    L ~ ln(N), where N is the number of categories.  Therefore, F.cross_entropy for a categorical should be scaled
    by ln(N).  Or for a single row:  F.cross_entropy(preds, y_true).item()/np.log(i)

    """
    def __init__(self, embed_dims, ohe_dims, n_numeric, dropout_noise, n_hidden):
        """

        :param embed_dims: {"column": (n_categories, embed_dim)}
        :param ohe_dims: {"col": ohe_size}
        :param n_numeric: number of numeric columns
        :param dropout_noise: dropout to apply to input
        :param n_hidden: number of units in dense hidden layer
        """
        super(DenoisingAE, self).__init__()
        self.embed_dims = embed_dims
        self.in_features = sum([v[1] for v in embed_dims.values()]) + sum(ohe_dims.values()) + n_numeric

        self.embeds = {k: nn.Embedding(v[0], v[1]) for k, v in embed_dims.items()}
        self.dropout = nn.Dropout(dropout_noise)
        self.dense = nn.Linear(self.in_features, n_hidden)

        out_map = {k: v[0] for k, v in self.embed_dims.items()}
        out_map.update(ohe_dims)

        self.out_map = out_map

        self.decoder = CategoricalDecoder(
            out_map,
            n_hidden,
        )

    def calc_loss(self, preds, ys):
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
        for i, n in enumerate(self.out_map.values()): # e.g. ("city", 1000)
            # Note:  averages the cross-entropy of column i for the whole batch
            loss = F.cross_entropy(preds[:,idx:n], ys[:,i], reduction='none') / th.log(th.FloatTensor([n]))
            #print(loss.shape)
            l.append(loss.reshape((-1,1)))
        return th.cat(l, dim=1).sum(dim=1).mean()

    def forward(self, inputs):
        embeds = list()
        for i, (k, v) in enumerate(self.embeds.items()):
            embeds.append(v(inputs[:,i].long()))  # Assumes first columns are embedding integers
        embeds.append(inputs[:, len(self.embeds):].float())  # the remaining inputs

        h = th.cat(embeds, dim=1)
        h = self.dropout(h)
        h = F.relu(self.dense(h))
        return self.decoder(h)


def train_ae(model, lr, wd, device):
    pass
