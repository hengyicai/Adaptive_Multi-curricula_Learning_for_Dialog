"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from parlai.agents.hy_lib.common_utils import texts_to_bow
import numpy as np
import math

from parlai.agents.adaptive_learning.helper import gVar, gData


class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidirectional, n_layers, noise_radius=0.2, dropout=0.2,
                 rnn_class='gru'):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dirs = 2 if self.bidirectional else 1
        assert type(self.bidirectional) == bool

        self.dropout = dropout
        self.embedding = embedder
        self.rnn_class = rnn_class
        if rnn_class == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_class == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise RuntimeError('RNN class {} is not supported yet!'.format(rnn_class))
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, inputs, input_lens=None, noise=False):
        if self.embedding is not None:
            inputs = self.embedding(inputs)

        batch_size, seq_len, emb_size = inputs.size()
        inputs = F.dropout(inputs, self.dropout, self.training)

        need_pack = False
        if input_lens is not None and need_pack:
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)

        init_hidden = gVar(torch.zeros(self.n_layers * (1 + self.bidirectional), batch_size, self.hidden_size))
        if self.rnn_class == 'lstm':
            init_hidden = (init_hidden, init_hidden)
        hids, h_n = self.rnn(inputs, init_hidden)
        if self.rnn_class == 'lstm':
            h_n = h_n[0]
        if input_lens is not None and need_pack:
            # noinspection PyUnboundLocalVariable
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1 + self.bidirectional), batch_size, self.hidden_size)
        h_n = h_n[-1]
        enc = h_n.transpose(1, 0).contiguous().view(batch_size, -1)
        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()), std=self.noise_radius))
            enc = enc + gauss_noise

        return enc, hids


class ContextEncoder(nn.Module):
    def __init__(self, utt_encoder, input_size, hidden_size, n_layers=1, noise_radius=0.2, rnn_class='gru'):
        super(ContextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.noise_radius = noise_radius

        self.n_layers = n_layers

        self.utt_encoder = utt_encoder
        self.rnn_class = rnn_class
        if self.rnn_class == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():  # initialize the gate weights with orthogonal
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, context, context_lens, utt_lens, floors, noise=False):
        batch_size, max_context_len, max_utt_len = context.size()
        utts = context.view(-1, max_utt_len)

        utt_lens = utt_lens.view(-1)
        utt_encs, _ = self.utt_encoder(utts, utt_lens)
        utt_encs = utt_encs.view(batch_size, max_context_len, -1)

        floor_one_hot = gVar(torch.zeros(floors.numel(), 2))
        floor_one_hot.data.scatter_(1, floors.view(-1, 1), 1)
        floor_one_hot = floor_one_hot.view(-1, max_context_len, 2)
        utt_floor_encs = torch.cat([utt_encs, floor_one_hot], 2)

        utt_floor_encs = F.dropout(utt_floor_encs, 0.25, self.training)
        context_lens_sorted, indices = context_lens.sort(descending=True)
        utt_floor_encs = utt_floor_encs.index_select(0, indices)
        utt_floor_encs = pack_padded_sequence(utt_floor_encs, context_lens_sorted.data.tolist(), batch_first=True)

        init_hidden = gVar(torch.zeros(1, batch_size, self.hidden_size))
        if self.rnn_class == 'lstm':
            hids, h_n = self.rnn(utt_floor_encs, (init_hidden, init_hidden))
            h_n = h_n[0]
        else:
            hids, h_n = self.rnn(utt_floor_encs, init_hidden)
        _, inv_indices = indices.sort()
        h_n = h_n.index_select(1, inv_indices)
        enc = h_n.transpose(1, 0).contiguous().view(batch_size, -1)

        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()), std=self.noise_radius))
            enc = enc + gauss_noise
        return enc


class Variation(nn.Module):
    def __init__(self, input_size, z_size):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu = nn.Linear(z_size, z_size)
        self.context_to_logsigma = nn.Linear(z_size, z_size)

        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size, _ = context.size()
        context = self.fc(context)
        mu = self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context)

        # mu = torch.clamp(mu, -30, 30)
        logsigma = torch.clamp(logsigma, -20, 20)
        std = torch.exp(0.5 * logsigma)
        epsilon = gVar(torch.randn([batch_size, self.z_size]))
        z = epsilon * std + mu
        return z, mu, logsigma


class MixVariation(nn.Module):
    def __init__(self, input_size, z_size, n_components, gumbel_temp=0.1):
        super(MixVariation, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.n_components = n_components
        self.gumbel_temp = 0.1

        self.pi_net = nn.Sequential(
            nn.Linear(z_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, n_components),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu = nn.Linear(z_size, n_components * z_size)
        self.context_to_logsigma = nn.Linear(z_size, n_components * z_size)
        self.pi_net.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size, _ = context.size()
        context = self.fc(context)

        pi = self.pi_net(context)
        pi = F.gumbel_softmax(pi, tau=self.gumbel_temp, hard=True, eps=1e-10)
        pi = pi.unsqueeze(1)

        mus = self.context_to_mu(context)
        logsigmas = self.context_to_logsigma(context)

        # mus = torch.clamp(mus, -30, 30)
        logsigmas = torch.clamp(logsigmas, -20, 20)

        stds = torch.exp(0.5 * logsigmas)

        epsilons = gVar(torch.randn([batch_size, self.n_components * self.z_size]))

        zi = (epsilons * stds + mus).view(batch_size, self.n_components, self.z_size)
        z = torch.bmm(pi, zi).squeeze(1)  # [batch_sz x z_sz]
        mu = torch.bmm(pi, mus.view(batch_size, self.n_components, self.z_size))
        logsigma = torch.bmm(pi, logsigmas.view(batch_size, self.n_components, self.z_size))
        return z, mu, logsigma


class Decoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, vocab_size,
                 n_layers=1, dropout=0.2, rnn_class='gru'):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = embedder
        self.rnn_class = rnn_class
        if self.rnn_class == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.fill_(0)

    def forward(self, init_hidden, context=None, inputs=None, lens=None):
        batch_size, maxlen = inputs.size()
        if self.embedding is not None:
            inputs = self.embedding(inputs)
        if context is not None:
            repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1)
            inputs = torch.cat([inputs, repeated_context], 2)
        inputs = F.dropout(inputs, self.dropout, self.training)
        init_hidden = init_hidden.unsqueeze(0)
        if self.rnn_class == 'lstm':
            init_hidden = (init_hidden, init_hidden)
        hids, h_n = self.rnn(inputs, init_hidden)
        decoded = self.out(hids.contiguous().view(-1, self.hidden_size))  # reshape before linear over vocab
        decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded

    def sampling(self, init_hidden, context, maxlen, SOS_tok, EOS_tok, mode='greedy'):
        batch_size = init_hidden.size(0)
        decoded_words = np.zeros((batch_size, maxlen), dtype=np.int)
        sample_lens = np.zeros(batch_size, dtype=np.int)

        decoder_input = gVar(torch.LongTensor([[SOS_tok] * batch_size]).view(batch_size, 1))
        decoder_input = self.embedding(decoder_input) if self.embedding is not None else decoder_input
        decoder_input = torch.cat([decoder_input, context.unsqueeze(1)], 2) if context is not None else decoder_input
        decoder_hidden = init_hidden.unsqueeze(0)
        if self.rnn_class == 'lstm':
            decoder_hidden = (decoder_hidden, decoder_hidden)
        for di in range(maxlen):
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            decoder_output = self.out(decoder_output)
            if mode == 'greedy':
                topi = decoder_output[:, -1].max(1, keepdim=True)[1]
            elif mode == 'sample':
                topi = torch.multinomial(F.softmax(decoder_output[:, -1], dim=1), 1)
            # noinspection PyUnboundLocalVariable
            decoder_input = self.embedding(topi) if self.embedding is not None else topi
            decoder_input = torch.cat([decoder_input, context.unsqueeze(1)],
                                      2) if context is not None else decoder_input
            ni = topi.squeeze().data.cpu().numpy()
            decoded_words[:, di] = ni

        for i in range(batch_size):
            for word in decoded_words[i]:
                if word == EOS_tok:
                    break
                sample_lens[i] = sample_lens[i] + 1
        return decoded_words, sample_lens


one = gData(torch.FloatTensor([1]))
minus_one = one * -1


class DialogWAE(nn.Module):
    def __init__(self, config, vocab_size, PAD_token=0, use_cuda=True, special_tokens=None):
        super(DialogWAE, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = config['maxlen']
        self.clip = config['clip']
        self.lambda_gp = config['lambda_gp']
        self.temp = config['temp']
        self.PAD_token = PAD_token
        self.special_tokens = special_tokens
        self.use_cuda = use_cuda
        self.embedder = nn.Embedding(vocab_size, config['emb_size'], padding_idx=PAD_token)
        self.utt_encoder = Encoder(self.embedder, config['emb_size'], config['n_hidden'], True,
                                   config['n_layers'], config['noise_radius'], config['dropout'],
                                   config.get('rnn_class', 'gru'))
        self.context_encoder = ContextEncoder(self.utt_encoder, config['n_hidden'] * 2 + 2, config['n_hidden'], 1,
                                              config['noise_radius'], config.get('rnn_class', 'gru'))
        self.prior_net = Variation(config['n_hidden'], config['z_size'])  # p(e|c)
        self.post_net = Variation(config['n_hidden'] * 3, config['z_size'])  # q(e|c,x)

        self.post_generator = nn.Sequential(
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size'])
        )
        self.post_generator.apply(self.init_weights)

        self.prior_generator = nn.Sequential(
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size'])
        )
        self.prior_generator.apply(self.init_weights)
        if config.get('hred', False):
            decoder_hidden_size = config['n_hidden']
        else:
            decoder_hidden_size = config['n_hidden'] + config['z_size']

        if config.get('vhred', False):
            self.vhred_priori = Variation(config['n_hidden'], config['z_size'])
            self.vhred_bow_project = nn.Sequential(
                nn.Linear(config['n_hidden'] + config['z_size'], config['n_hidden'] * 2),
                nn.BatchNorm1d(config['n_hidden'] * 2, eps=1e-05, momentum=0.1),
                nn.Tanh(),
                nn.Linear(config['n_hidden'] * 2, config['n_hidden'] * 2),
                nn.BatchNorm1d(config['n_hidden'] * 2, eps=1e-05, momentum=0.1),
                nn.Tanh(),
                nn.Linear(config['n_hidden'] * 2, vocab_size)
            )
            self.vhred_posterior = Variation(config['n_hidden'] * 3, config['z_size'])

        self.decoder = Decoder(self.embedder, config['emb_size'], decoder_hidden_size,
                               vocab_size, n_layers=1, dropout=config['dropout'],
                               rnn_class=config.get('rnn_class', 'gru'))

        self.discriminator = nn.Sequential(
            nn.Linear(config['n_hidden'] + config['z_size'], config['n_hidden'] * 2),
            nn.BatchNorm1d(config['n_hidden'] * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config['n_hidden'] * 2, config['n_hidden'] * 2),
            nn.BatchNorm1d(config['n_hidden'] * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config['n_hidden'] * 2, 1),
        )
        self.discriminator.apply(self.init_weights)
        self.config = config
        # self.criterion_ce = nn.CrossEntropyLoss()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def sample_code_post(self, x, c):
        e, _, _ = self.post_net(torch.cat((x, c), 1))
        z = self.post_generator(e)
        return z

    def sample_code_prior(self, c):
        e, _, _ = self.prior_net(c)
        z = self.prior_generator(e)
        return z

    def normal_kl_div(self, mean1, logvar1, mean2=None, logvar2=None):
        if mean2 is None:
            mean2 = Variable(torch.FloatTensor([0.0])).unsqueeze(dim=1).expand(mean1.size(0), mean1.size(1))
            if self.use_cuda:
                mean2 = mean2.cuda()
        if logvar2 is None:
            logvar2 = Variable(torch.FloatTensor([0.0])).unsqueeze(dim=1).expand(logvar1.size(0), logvar1.size(1))
            if self.use_cuda:
                logvar2 = logvar2.cuda()
        kl_div = 0.5 * torch.sum(
            logvar2 - logvar1 + (torch.exp(logvar1) + (mean1 - mean2).pow(2)) / torch.exp(logvar2) - 1.0,
            dim=1).mean().squeeze()
        return kl_div

    def compute_loss_AE(self, context, context_lens, utt_lens, floors, response, res_lens):
        c = self.context_encoder(context, context_lens, utt_lens, floors)

        vhred_kl_loss = -1
        bow_loss = -1

        if self.config.get('hred', False):
            output = self.decoder(c, None, response[:, :-1], (res_lens - 1))
        elif self.config.get('vhred', False):
            x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
            vhred_post_z, vhred_post_mu, vhred_post_logsigma = self.vhred_posterior(torch.cat((x, c), 1))
            vhred_priori_z, vhred_priori_mu, vhred_priori_logsigma = self.vhred_priori(c)
            output = self.decoder(torch.cat((vhred_post_z, c), 1), None, response[:, :-1], (res_lens - 1))
            vhred_kl_loss = self.normal_kl_div(vhred_post_mu, vhred_post_logsigma, vhred_priori_mu,
                                               vhred_priori_logsigma)

            bow_logits = self.vhred_bow_project(torch.cat([c, vhred_priori_z], dim=1))
            bow_loss = self._compute_bow_loss(bow_logits, response)
        else:
            x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
            z = self.sample_code_post(x, c)
            c_z = torch.cat((z, c), 1)
            if self.config.get('norm_z', False):
                c_z = F.layer_norm(c_z, (c_z.size(-1),))
            output = self.decoder(c_z, None, response[:, :-1], (res_lens - 1))

        _, preds = output.max(dim=2)

        flattened_output = output.view(-1, self.vocab_size)

        notnull = response[:, :-1].ne(self.PAD_token)
        target_tokens = notnull.long().sum().item()
        correct = ((response[:, :-1] == preds) * notnull).sum().item()

        dec_target = response[:, 1:].contiguous().view(-1)
        # mask = dec_target.gt(0)  # [(batch_sz*seq_len)]
        # masked_target = dec_target.masked_select(mask)
        masked_target = dec_target
        # output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz*seq_len) x n_tokens]
        # masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        masked_output = flattened_output

        return {'masked_output': masked_output, 'masked_target': masked_target, 'target_tokens': target_tokens,
                'correct_tokens': correct, 'num_tokens': target_tokens, 'scores': output, 'preds': preds,
                'vhred_kl_loss': vhred_kl_loss, 'bow_loss': bow_loss}

    @staticmethod
    def anneal_weight(step):
        return (math.tanh((step - 3500) / 1000) + 1) / 2

    def _compute_bow_loss(self, bow_logits, response):
        target_bow = texts_to_bow(response, self.vocab_size, self.special_tokens)
        if self.use_cuda:
            target_bow = target_bow.cuda()
        bow_loss = -F.log_softmax(bow_logits, dim=1) * target_bow
        # Compute per token loss
        # bow_loss = torch.sum(bow_loss) / torch.sum(target_bow)
        bow_loss = torch.sum(bow_loss) / response.size(0)
        return bow_loss

    def train_AE(self, context, context_lens, utt_lens, floors, response, res_lens, optimizer_AE, criterion_ce,
                 steps=0):
        # self.context_encoder.train()
        # self.decoder.train()

        loss_dict = self.compute_loss_AE(context, context_lens, utt_lens, floors, response, res_lens)
        masked_output = loss_dict['masked_output']
        masked_target = loss_dict['masked_target']
        target_tokens = loss_dict['target_tokens']
        correct = loss_dict['correct_tokens']
        output = loss_dict['scores']
        preds = loss_dict['preds']
        vhred_kl_loss = loss_dict['vhred_kl_loss']
        bow_loss = loss_dict['bow_loss']

        optimizer_AE.zero_grad()
        loss = criterion_ce(masked_output / self.temp, masked_target)
        sum_loss = loss.item()
        loss /= target_tokens
        avg_loss = loss.item()

        params_to_clip = list(self.context_encoder.parameters()) + list(self.decoder.parameters())
        if vhred_kl_loss != -1 and self.config.get('vhred', False) \
                and not (torch.isnan(vhred_kl_loss) or torch.isinf(vhred_kl_loss)) \
                and not (torch.isnan(bow_loss) or torch.isinf(bow_loss)):
            loss += (vhred_kl_loss * self.anneal_weight(steps) + 0.01 * bow_loss)
            vhred_kl_loss = vhred_kl_loss.item()
            bow_loss = bow_loss.item()
            params_to_clip = params_to_clip + list(self.vhred_bow_project.parameters()) + list(
                self.vhred_priori.parameters()) + list(self.vhred_posterior.parameters())
        loss.backward()

        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(params_to_clip, self.clip)
        optimizer_AE.step()

        return {'nll_loss': sum_loss, 'avg_loss': avg_loss, 'correct_tokens': correct,
                'num_tokens': target_tokens, 'scores': output, 'preds': preds,
                'vhred_kl_loss': vhred_kl_loss, 'bow_loss': bow_loss}

    def train_G(self, context, context_lens, utt_lens, floors, response, res_lens, optimizer_G):
        # self.context_encoder.eval()
        optimizer_G.zero_grad()

        for p in self.discriminator.parameters():
            p.requires_grad = False

        with torch.no_grad():
            c = self.context_encoder(context, context_lens, utt_lens, floors)
            x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
        # -----------------posterior samples ---------------------------
        z_post = self.sample_code_post(x.detach(), c.detach())

        errG_post = torch.mean(self.discriminator(torch.cat((z_post, c.detach()), 1)))
        errG_post.backward(minus_one)

        # ----------------- prior samples ---------------------------
        prior_z = self.sample_code_prior(c.detach())
        errG_prior = torch.mean(self.discriminator(torch.cat((prior_z, c.detach()), 1)))
        errG_prior.backward(one)

        optimizer_G.step()

        for p in self.discriminator.parameters():
            p.requires_grad = True

        costG = errG_prior - errG_post
        return {'train_loss_G': costG.item()}

    def train_D(self, context, context_lens, utt_lens, floors, response, res_lens, optimizer_D):
        # self.context_encoder.eval()
        self.discriminator.train()

        optimizer_D.zero_grad()

        batch_size = context.size(0)

        with torch.no_grad():
            c = self.context_encoder(context, context_lens, utt_lens, floors)
            x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)

        post_z = self.sample_code_post(x, c)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z.detach(), c.detach()), 1)))
        errD_post.backward(one)

        prior_z = self.sample_code_prior(c)
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z.detach(), c.detach()), 1)))
        errD_prior.backward(minus_one)

        alpha = gData(torch.rand(batch_size, 1))
        alpha = alpha.expand(prior_z.size())
        interpolates = alpha * prior_z.data + ((1 - alpha) * post_z.data)
        interpolates = Variable(interpolates, requires_grad=True)
        d_input = torch.cat((interpolates, c.detach()), 1)
        disc_interpolates = torch.mean(self.discriminator(d_input))
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=gData(torch.ones(disc_interpolates.size())),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.contiguous().view(gradients.size(0), -1).norm(2,
                                                                                     dim=1) - 1) ** 2).mean() * self.lambda_gp
        gradient_penalty.backward()

        optimizer_D.step()
        costD = -(errD_prior - errD_post) + gradient_penalty
        return {'train_loss_D': costD.item()}

    def valid(self, context, context_lens, utt_lens, floors, response, res_lens, criterion_ce):
        # self.context_encoder.eval()
        # self.discriminator.eval()
        # self.decoder.eval()

        c = self.context_encoder(context, context_lens, utt_lens, floors)

        if not self.config.get('hred', False) and not self.config.get('vhred', False):
            x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
            post_z = self.sample_code_post(x, c)
            prior_z = self.sample_code_prior(c)
            errD_post = torch.mean(self.discriminator(torch.cat((post_z, c), 1)))
            errD_prior = torch.mean(self.discriminator(torch.cat((prior_z, c), 1)))
            costD = -(errD_prior - errD_post)
            costG = -costD

        dec_target = response[:, 1:].contiguous().view(-1)
        mask = dec_target.gt(0)  # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask)
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)

        vhred_kl_loss = -1
        bow_loss = -1
        if self.config.get('hred', False):
            dec_input = c
        elif self.config.get('vhred', False):
            x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
            vhred_post_z, vhred_post_mu, vhred_post_logsigma = self.vhred_posterior(torch.cat((x, c), 1))
            vhred_priori_z, vhred_priori_mu, vhred_priori_logsigma = self.vhred_priori(c)
            vhred_kl_loss = self.normal_kl_div(vhred_post_mu, vhred_post_logsigma, vhred_priori_mu,
                                               vhred_priori_logsigma)
            dec_input = torch.cat((vhred_post_z, c), 1)

            bow_logits = self.vhred_bow_project(torch.cat([c, vhred_priori_z], dim=1))
            bow_loss = self._compute_bow_loss(bow_logits, response)

            vhred_kl_loss = vhred_kl_loss.item()
            bow_loss = bow_loss.item()

        else:
            # noinspection PyUnboundLocalVariable
            dec_input = torch.cat((post_z, c), 1)
            if self.config.get('norm_z', False):
                dec_input = F.layer_norm(dec_input, (dec_input.size(-1),))

        output = self.decoder(dec_input, None, response[:, :-1], (res_lens - 1))

        _, preds = output.max(dim=2)

        flattened_output = output.view(-1, self.vocab_size)

        notnull = response[:, :-1].ne(self.PAD_token)
        target_tokens = notnull.long().sum().item()
        correct = ((response[:, :-1] == preds) * notnull).sum().item()

        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        lossAE = criterion_ce(masked_output / self.temp, masked_target)
        sum_loss = lossAE.item()
        lossAE /= target_tokens
        avg_loss = lossAE.item()

        valid_loss = {
            'nll_loss': sum_loss, 'avg_loss': avg_loss, 'vhred_kl_loss': vhred_kl_loss,
            'correct_tokens': correct, 'num_tokens': target_tokens, 'bow_loss': bow_loss,
        }
        if not self.config.get('hred', False) and not self.config.get('vhred', False):
            # noinspection PyUnboundLocalVariable
            valid_loss['valid_loss_G'] = costG.item()
            # noinspection PyUnboundLocalVariable
            valid_loss['valid_loss_D'] = costD.item()
        return valid_loss

    def sample(self, context, context_lens, utt_lens, floors, SOS_tok, EOS_tok):
        # self.context_encoder.eval()
        # self.decoder.eval()

        c = self.context_encoder(context, context_lens, utt_lens, floors)
        if self.config.get('hred', False):
            dec_input = c
        elif self.config.get('vhred', False):
            prior_z, _, _ = self.vhred_priori(c)
            dec_input = torch.cat((prior_z, c), 1)
        else:
            prior_z = self.sample_code_prior(c)
            dec_input = torch.cat((prior_z, c), 1)
            if self.config.get('norm_z', False):
                dec_input = F.layer_norm(dec_input, (dec_input.size(-1),))

        sample_words, sample_lens = self.decoder.sampling(
            dec_input, None, self.maxlen, SOS_tok, EOS_tok, "greedy")
        return sample_words, sample_lens


class DialogWAE_GMP(DialogWAE):
    def __init__(self, config, vocab_size, PAD_token=0, use_cuda=True, special_tokens=None):
        super(DialogWAE_GMP, self).__init__(config, vocab_size, PAD_token, use_cuda, special_tokens)
        self.n_components = config['n_prior_components']
        self.gumbel_temp = config['gumbel_temp']

        self.prior_net = MixVariation(config['n_hidden'], config['z_size'], self.n_components,
                                      self.gumbel_temp)  # p(e|c)
