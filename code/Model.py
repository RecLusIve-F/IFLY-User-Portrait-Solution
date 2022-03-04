import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SelfAttentionDot(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(SelfAttentionDot, self).__init__()

        self.map_2_u = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.context_u = nn.Parameter(torch.FloatTensor(attention_dim, 1))
        nn.init.uniform_(self.context_u)

    def forward(self, hidden_state):
        """
        :param hidden_state: batch_size * n * d
        :return: batch_size * d
        """
        u_mat = self.map_2_u(hidden_state)
        batched_context_u = torch.stack([self.context_u] * hidden_state.shape[0], dim=0)
        alignment = torch.bmm(u_mat, batched_context_u)
        alignment = F.softmax(alignment, dim=1)
        result = torch.bmm(alignment.transpose(1, 2), hidden_state)
        return result.squeeze(1)


class Model(nn.Module):
    def __init__(self, tagid_vocab_size, city_vocab_size, province_vocab_size, args):
        super(Model, self).__init__()

        tagid_embed_size, city_embed_size, province_embed_size, time_embed_size = args.embed_size
        dropout_rate = args.dropout

        self.tagid_embedder = nn.Embedding(tagid_vocab_size, tagid_embed_size)
        self.city_embedder = nn.Embedding(city_vocab_size, city_embed_size)
        self.province_embedder = nn.Embedding(province_vocab_size, province_embed_size)
        self.time_layer_embedding = nn.Linear(1, time_embed_size)

        self.tagid_f_gru = nn.GRU(tagid_embed_size, 128, num_layers=1, bidirectional=True, batch_first=True)
        self.tagid_b_gru = nn.GRU(tagid_embed_size, 128, num_layers=1, bidirectional=True, batch_first=True)

        self.tagid_f_dense = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(tagid_embed_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.tagid_b_dense = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(tagid_embed_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.tagid_f_att_concat = SelfAttentionDot(512, 512)
        self.tagid_b_att_concat = SelfAttentionDot(512, 152)
        self.tagid_f_att_gru = SelfAttentionDot(256, 256)
        self.tagid_b_att_gru = SelfAttentionDot(256, 256)

        self.tagid_final_dense = nn.Linear(3584, 1024)

        self.time_f_gru = nn.GRU(time_embed_size, 128, num_layers=1, bidirectional=True, batch_first=True)
        self.time_b_gru = nn.GRU(time_embed_size, 128, num_layers=1, bidirectional=True, batch_first=True)

        self.time_f_att = SelfAttentionDot(256, 256)
        self.time_b_att = SelfAttentionDot(256, 256)

        self.time_final_dense = nn.Linear(1536, 512)

        self.location_dense = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(78, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.feature_dense = nn.Sequential(
            nn.BatchNorm1d(346),
            nn.Linear(346, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.classifier_dense = nn.Sequential(
            nn.BatchNorm1d(1664),
            nn.Dropout(dropout_rate),
            nn.Linear(1664, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def _conv_and_pool(self, x):
        x1 = F.max_pool1d(x, x.size(2)).squeeze(dim=2)
        x2 = F.avg_pool1d(x, x.size(2)).squeeze(dim=2)
        return torch.cat([x1, x2], dim=1)

    def forward(self, tagid_f, tagid_b, cityid, provinceid, time_f, time_b, onehot, pid_tagid_change_time):
        tagid_f_embedding = self.tagid_embedder(tagid_f)
        tagid_b_embedding = self.tagid_embedder(tagid_b)

        city_embedding = self.city_embedder(cityid)
        province_embedding = self.province_embedder(provinceid)
        location_embedding = torch.cat([province_embedding, city_embedding], dim=1)

        time_f_embedding = self.time_layer_embedding(time_f.unsqueeze(-1))
        time_b_embedding = self.time_layer_embedding(time_b.unsqueeze(-1))

        tagid_f_gru_embedding, _ = self.tagid_f_gru(tagid_f_embedding)
        tagid_b_gru_embedding, _ = self.tagid_b_gru(tagid_b_embedding)

        tagid_f_dense_embedding = self.tagid_f_dense(tagid_f_embedding)
        tagid_b_dense_embedding = self.tagid_b_dense(tagid_b_embedding)

        tagid_f_concat_embedding = torch.cat([tagid_f_gru_embedding, tagid_f_dense_embedding], dim=2)
        tagid_b_concat_embedding = torch.cat([tagid_b_gru_embedding, tagid_b_dense_embedding], dim=2)

        tagid_f_att_embedding = self.tagid_f_att_concat(tagid_f_concat_embedding)
        tagid_b_att_embedding = self.tagid_b_att_concat(tagid_b_concat_embedding)

        tagid_f_att_gru_embedding = self.tagid_f_att_gru(tagid_f_gru_embedding)
        tagid_b_att_gru_embedding = self.tagid_b_att_gru(tagid_b_gru_embedding)

        tagid_f_pool_embedding = self._conv_and_pool(tagid_f_concat_embedding.permute(0, 2, 1).contiguous())
        tagid_b_pool_embedding = self._conv_and_pool(tagid_b_concat_embedding.permute(0, 2, 1).contiguous())

        tagid_final_embedding = torch.cat(
            [tagid_f_att_embedding, tagid_f_pool_embedding, tagid_f_att_gru_embedding, tagid_b_att_embedding,
             tagid_b_pool_embedding, tagid_b_att_gru_embedding], dim=1)

        tagid_final_embedding = self.tagid_final_dense(tagid_final_embedding)

        time_f_gru_embedding, _ = self.time_f_gru(time_f_embedding)
        time_b_gru_embedding, _ = self.time_b_gru(time_b_embedding)

        time_f_att_embedding = self.time_f_att(time_f_gru_embedding)
        time_b_att_embedding = self.time_b_att(time_b_gru_embedding)

        time_f_pool_embedding = self._conv_and_pool(time_f_gru_embedding.permute(0, 2, 1).contiguous())
        time_b_pool_embedding = self._conv_and_pool(time_b_gru_embedding.permute(0, 2, 1).contiguous())

        time_final_embedding = torch.cat(
            [time_f_att_embedding, time_f_pool_embedding, time_b_att_embedding, time_b_pool_embedding], dim=1)

        time_final_embedding = self.time_final_dense(time_final_embedding)

        location_final_embedding = self.location_dense(location_embedding)

        fearure_embedding = torch.cat(
            (location_final_embedding, time_f.float(), time_b.float(), onehot.float(), pid_tagid_change_time.float()),
            dim=1)
        feature_final_embedding = self.feature_dense(fearure_embedding)

        final_embedding = torch.cat([tagid_final_embedding, feature_final_embedding, time_final_embedding], dim=1)
        output = self.classifier_dense(final_embedding)

        return output
