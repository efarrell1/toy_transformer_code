# @title

# Imports packages and installs transformerlens and circuitsvis

# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False

IN_COLAB = False
if IN_COLAB:
    import google.colab
    IN_COLAB = True
    # print("Running as a Colab notebook")
    # %pip install git+https://github.com/neelnanda-io/TransformerLens.git
    # %pip install circuitsvis
else:
    IN_COLAB = False
    import plotly.io as pio
    pio.renderers
    pio.renderers.default = "vscode"
    # print("Running as a Jupyter notebook - intended for development only!")
    import plotly.io as pio
    pio.renderers.default = "notebook_connected"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader
import plotly.graph_objects as go

from jaxtyping import Float, Int
from typing import List, Union, Optional
from functools import partial
import copy

from scipy.fft import fft, ifft, fftfreq
import pandas as pd

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import circuitsvis as cv

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)
# Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from einops import reduce, rearrange, repeat

torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"


import logging
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, overload

import einops
import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm
from fancy_einsum import einsum
from jaxtyping import Float, Int
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing_extensions import Literal


import transformer_lens.loading_from_pretrained as loading
import transformer_lens.utils as utils
from transformer_lens import HookedTransformerConfig
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import (
    Embed,
    LayerNorm,
    LayerNormPre,
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    TransformerBlock,
    Unembed,
)
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint

# Note - activation cache is used with run_with_cache, past_key_value_caching is used for generation.
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utilities import devices
# from transformer_lens.utils import USE_DEFAULT_VALUE
from transformer_lens.utilities import devices

import transformer_lens.patching as patching



# @title
# Some plotting routines

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


# Functions to generate randomised token patterns
def generate_repeated_tokens(model, seq_len, batch):
    '''
    Generates a sequence of repeated random tokens
    of the form: a, b, c, x, b, y, a, b, c, x, b, y

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    # SOLUTION
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence[:, -2] = first_sequence[:, -2]
    # We want a sequence of the form: a, b, c, x, b, y, a, b, c, x, b, y
    token_sequence = torch.cat([prefix, first_sequence, second_sequence, first_sequence, second_sequence], dim=-1).to(device)
    return token_sequence

triplet_str_tokens = [' ', ' Alpha', ' Beta', ' Gamma', ' Chi', ' Beta', ' Epsilon', ' Alpha', ' Beta', ' Gamma', ' Chi', ' Beta', ' Epsilon', ' Alpha', ' Beta', ' Gamma', ' Chi', ' Beta', ' Epsilon']


def generate_tokens_quad_seq(model, seq_len, batch):
    '''
    Generates a sequence of repeated random tokens
    of the form: a, b, c, d, x, b, c, y, a, b, c, d, x, b, c, y

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    # SOLUTION
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence[:, -2] = first_sequence[:, -2]
    second_sequence[:, -3] = first_sequence[:, -3]
    # We want a sequence of the form: a, b, c, d, x, b, c, y, a, b, c, d, x, b, c, y
    token_sequence = torch.cat([prefix, first_sequence, second_sequence, first_sequence, second_sequence], dim=-1).to(device)
    return token_sequence

quad_str_tokens = [' ', ' Alpha', ' Beta', ' Gamma', ' Delta', ' Chi', ' Beta', ' Gamma', ' Epsilon',
                   ' Alpha', ' Beta', ' Gamma', ' Delta', ' Chi', ' Beta', ' Gamma', ' Epsilon']

def generate_tokens_abac_seq(model, seq_len, batch):
    '''
    Generates a sequence of repeated random tokens
    of the form: a, b, x, x, a, c, x, x, a

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    seq_len = 13 + 3
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    first_sequence[:, 6 + 3] = first_sequence[:, 1]
    first_sequence[:, 11 + 3] = first_sequence[:, 1]
    # We want a sequence of the form: a, b, x, x, a, c, x, x, a
    token_sequence = torch.cat([prefix, first_sequence], dim=-1).to(device)
    return token_sequence

abac_str_tokens = [' ', ' Random', ' Alpha', ' Beta', ' Random', ' Random', 'Random', ' Random', ' Random', 'Random', ' Alpha', ' Gamma', ' Random', ' Random', ' Random', ' Alpha', ' Random']


def generate_random_token_sequence(model, seq_len, batch, prefix_flag=True):
    '''
    Generates a sequence of random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    # SOLUTION
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    if prefix_flag:
      token_sequence = torch.cat([prefix, first_sequence], dim=-1).to(device)
    else:
      token_sequence = torch.cat([first_sequence], dim=-1).to(device)
    return token_sequence


def generate_random_same_word(model, seq_len, batch, prefix_flag=False):
    '''
    Generates a sequence of repeated random tokens
    of the form: with a constant word in there somewhere

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    # SOLUTION
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, 1), dtype=torch.int64)
    first_sequence = repeat(first_sequence, 'b s -> b (repeat s)', repeat=seq_len)
    if prefix_flag:
      token_sequence = torch.cat([prefix, first_sequence], dim=-1).to(device)
    else:
      token_sequence = torch.cat([first_sequence], dim=-1).to(device)
    return token_sequence


def get_attn_patterns(cache):
  nlayers = len([x for x in list(cache.keys()) if x.endswith(".attn.hook_pattern")])
  attn_patterns = []

  for layer in range(nlayers):
    batch_attn_pattern = cache["pattern", layer, "attn"]
    attn_pattern = reduce(batch_attn_pattern, 'b h w c -> h w c', 'mean')

    attn_patterns.append(attn_pattern)

  return attn_patterns

def apply_causal_mask(attn_scores):
    '''
    Applies a causal mask to attention scores, and returns masked scores.
    '''
    # SOLUTION
    # Define a mask that is True for all positions we want to set probabilities to zero for
    all_ones = torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
    mask = torch.triu(all_ones, diagonal=1).bool()
    # Apply the mask to attention scores, then return the masked scores
    attn_scores.masked_fill_(mask, -1e5)
    return attn_scores


def apply_causal_mask1d(attn_scores):
    '''
    Applies a causal mask to attention scores, and returns masked scores.
    '''
    # SOLUTION
    # Define a mask that is True for all positions we want to set probabilities to zero for
    all_ones = torch.ones(attn_scores.size(-1), device=attn_scores.device)
    mask = torch.triu(all_ones, diagonal=1).bool()
    # Apply the mask to attention scores, then return the masked scores
    attn_scores.masked_fill_(mask, -1e5)
    return attn_scores


# def generate_random_with_word(model, seq_len, batch, word='and'):
#     '''
#     Generates a sequence of repeated random tokens
#     of the form: with a constant word in there somewhere

#     Outputs are:
#         rep_tokens: [batch, 1+2*seq_len]
#     '''
#     # SOLUTION
#     prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
#     first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, 1), dtype=torch.int64)
#     first_sequence =
#     # We want a sequence of the form: a, b, c, x, b, y, a, b, c, x, b, y
#     token_sequence = torch.cat([prefix, first_sequence], dim=-1).to(device)
#     return token_sequence


# @title

def generate_random_same_word(model, seq_len, batch, prefix_flag=False):
    '''
    Generates a sequence of repeated random tokens
    of the form: with a constant word in there somewhere

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, 1), dtype=torch.int64)
    first_sequence = repeat(first_sequence, 'b s -> b (repeat s)', repeat=seq_len)
    if prefix_flag:
      token_sequence = torch.cat([prefix, first_sequence], dim=-1).to(device)
    else:
      token_sequence = torch.cat([first_sequence], dim=-1).to(device)
    return token_sequence


def generate_sequential(model, seq_len, batch, prefix_flag=False):
    '''
    Generates a sequence of repeated random tokens
    of the form: with a constant word in there somewhere

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    first_sequence = torch.from_numpy(np.arange(0, seq_len)).type(torch.int64)
    first_sequence = rearrange(first_sequence, 's -> 1 s')
    first_sequence = repeat(first_sequence, 'b s -> (repeat b) s', repeat=batch)
    if prefix_flag:
      token_sequence = torch.cat([prefix, first_sequence], dim=-1).to(device)
    else:
      token_sequence = torch.cat([first_sequence], dim=-1).to(device)
    return token_sequence


def generate_arr(model, array, batch, prefix_flag=False):
    '''
    Generates a sequence of repeated random tokens
    of the form: with a constant word in there somewhere

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    first_sequence = torch.from_numpy(array).type(torch.int64)
    first_sequence = rearrange(first_sequence, 's -> 1 s')
    first_sequence = repeat(first_sequence, 'b s -> (repeat b) s', repeat=batch)
    if prefix_flag:
      token_sequence = torch.cat([prefix, first_sequence], dim=-1).to(device)
    else:
      token_sequence = torch.cat([first_sequence], dim=-1).to(device)
    return token_sequence


def get_strings(model, x):
  return model.to_str_tokens(torch.from_numpy(np.array(x)))

# @title
def get_cosine_sims(w, resid, bias=0):
  a_norm = w / w.norm(dim=1)[:, None]
  b_norm = resid / resid.norm(dim=1)[:, None]
  res = torch.mm(a_norm, b_norm.transpose(0,1))

  softmax = nn.Softmax(dim=0)
  relu = nn.ReLU()

  res_softmax = softmax(relu(res - bias))

  return res, res_softmax

def get_token_value(resid):
  res, res_softmax = get_cosine_sims(w_tok_embed, resid, bias=0.2)

  imax = [np.argmax(x) for x in res_softmax.T.cpu()]

  most_likely_strings = get_strings(model, imax)

  return res, res_softmax, most_likely_strings


def get_position_value(resid, bias=0.3):
  res, res_softmax = get_cosine_sims(w_pos_embed, resid, bias=bias)

  imax = [np.argmax(x).item() for x in res_softmax.T.cpu()]

  return res, res_softmax, imax


# @title
def get_resid_at_layer(resid, nlayer=2):
  for i in range(nlayer):
    resid_pre = forward_layer_norm(resid, i)
    attn_pattern, attn_out = forward_attention_layer(resid_pre, i)
    resid = resid + attn_out
  return resid


def forward_layer_norm(resid_pre, i=0):

  scale = repeat(cache['blocks.' + str(i) + '.ln1.hook_scale'], 'b s o -> b s (repeat o)', repeat=resid_pre.shape[2])
  normalized_resid_pre = torch.div(resid_pre, scale)

  return normalized_resid_pre


def forward_attention_layer(normalized_resid_pre, i=0):

  q = einops.einsum(
      normalized_resid_pre,  model.W_Q[i],
      "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
  ) + model.b_Q[i]

  k = einops.einsum(
      normalized_resid_pre, model.W_K[i],
      "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
  ) + model.b_K[i]

  v = einops.einsum(
      normalized_resid_pre, model.W_V[i],
      "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
  ) + model.b_V[i]

  attn_scores = einops.einsum(
      q, k,
      "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
  )

  attn_scores_masked = apply_causal_mask(attn_scores / model.cfg.d_head ** 0.5)
  attn_pattern = attn_scores_masked.softmax(-1)

  # Take weighted sum of value vectors, according to attention probabilities
  z = einops.einsum(
      v, attn_pattern,
      "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
  )

  # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
  attn_out = einops.einsum(
      z, model.W_O[i],
      "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
  ) + model.b_O[i]

  return attn_pattern, attn_out, z


# @title
def get_attn_given_qk(qresid, kresid, W_Q, b_Q, W_K, b_K):
  q = einops.einsum(
    qresid,  W_Q,
    "d_model, d_model d_head -> d_head",
  ) + b_Q

  k = einops.einsum(
    W_K, kresid,
    "d_model d_head, pos d_model -> pos d_head",
  ) #+ b_K

  q_W_K = einops.einsum(
    q,  W_K,
    "d_head, d_model d_head -> d_model",
  ) / model.cfg.d_head ** 0.5

  scores = einops.einsum(
    q_W_K,  kresid,
    "d_model, pos d_model -> pos",
  )

  return scores, q_W_K


def get_attn_given_qk_batch(qresid, kresid, W_Q, b_Q, W_K, b_K):
  q = einops.einsum(
    qresid,  W_Q,
    "batch d_model, d_model d_head -> batch d_head",
  ) + b_Q

  k = einops.einsum(
    W_K, kresid,
    "d_model d_head, pos d_model -> pos d_head",
  ) #+ b_K

  q_W_K = einops.einsum(
    q,  W_K,
    "batch d_head, d_model d_head -> batch d_model",
  ) / model.cfg.d_head ** 0.5

  scores = einops.einsum(
    q_W_K,  kresid,
    " batch d_model, pos d_model -> batch pos",
  )

  return scores, q_W_K


def get_ov_given_qk_batch(resid, W_V, b_V, W_O, b_O):
  v = einops.einsum(
    resid,  W_V,
    "batch d_model, d_model d_head -> batch d_head",
  ) + b_V

  out = einops.einsum(
      v, W_O,
      "batch d_head, d_head d_model -> batch d_model"
  ) + b_O

  return out

def generate_first_n(model, batch, prefix_flag=True):
  '''
  random
  '''
  prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
  arr = np.column_stack((np.full(batch, 1130), np.full(batch, 2350), np.arange(batch)))
  sequence = torch.from_numpy(arr).type(torch.int64)
  if prefix_flag:
    token_sequence = torch.cat([prefix, sequence], dim=-1).to(device)
  else:
    token_sequence = torch.cat([sequence], dim=-1).to(device)
  return token_sequence


def attn_pattern_from_string(input_string):
  tokens = model.to_tokens(input_string)
  str_tokens = model.to_str_tokens(input_string)

  logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
  attn_patterns = get_attn_patterns(cache)

  return str_tokens, attn_patterns

def plot_apfs(input_string, i=0):
  str_tokens, attn_patterns = attn_pattern_from_string(input_string)
  return {'tokens':str_tokens, 'attention':attn_patterns[i]}


def get_attn_scores(cache):
  nlayers = len([x for x in list(cache.keys()) if x.endswith(".attn.hook_attn_scores")])
  attn_patterns = []

  for layer in range(nlayers):
    batch_attn_pattern = cache["attn_scores", layer, "attn"]
    attn_pattern = reduce(batch_attn_pattern, 'b h w c -> h w c', 'mean')

    attn_patterns.append(attn_pattern)

  return attn_patterns


def fig_to_json(fig, json_dir, name):
    fig.write_json(json_dir + "/" + name + ".json")
    return
json_dir = '../blog/static'

def cv_fig_to_html(fig, html_dir, name):
   with open(html_dir + "/" + name + ".html", "w") as f:
       f.write(fig.__str__())
html_dir = '../blog/layouts/shortcodes'

# @title
class CustomHookedTransformer(HookedTransformer):
  def __init__(self, cfg, tokenizer=None, move_to_device=True, default_padding_side='right'):
    super().__init__(cfg, tokenizer, move_to_device, default_padding_side)
    self.cfg.custom_type = 'None'

  # TODO make sure type assertions are provided
  def forward(
      self,
      input,
      return_type = "logits",
      loss_per_token = False,
      prepend_bos = None,
      padding_side = None,
      stop_at_layer = None,
      past_kv_cache = None,
      past_left_attention_mask = None,  # [batch pos]
  ):

    with utils.LocallyOverridenDefaults(
        self, prepend_bos=prepend_bos, padding_side=padding_side
    ):
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # This is only intended to support passing in a single string
            tokens = self.to_tokens(
                input, prepend_bos=prepend_bos, padding_side=padding_side
            )
        else:
            tokens = input
        if len(tokens.shape) == 1:
            # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
            tokens = tokens[None]
        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(devices.get_device_for_block_index(0, self.cfg))

        if self.tokenizer and self.tokenizer.padding_side == "left":
            # If the padding side is left, we need to compute the attention mask for the adjustment of
            # absolute positional embeddings and attention masking so that the pad tokens are not attended.

            if past_left_attention_mask is None:
                left_attention_mask = utils.get_attention_mask(
                    self.tokenizer, tokens, self.cfg.default_prepend_bos
                )
            else:
                assert (
                    past_kv_cache is not None
                ), "If past_left_attention_mask is not None, past_kv_cache must not be None"
                assert (
                    tokens.shape[1] == 1
                ), "If past_left_attention_mask is not None, tokens must be a single token along the sequence dimension"
                # past_kv_cache is not None, so we're doing caching.
                # We need to extend the past_left_attention_mask.
                # Append '1' to the right of the past_left_attention_mask to account for the new tokens
                left_attention_mask = utils.extend_tensor_with_ones(
                    past_left_attention_mask
                )

        else:
            # If tokenizer is not set, we assume that the input is right-padded.
            # If the padding side is right, we don't need to compute the attention mask.
            # We separate this case from left padding for computational efficiency.
            left_attention_mask = None

        # If we're doing caching, then we reuse keys and values from previous runs, as that's the only
        # way that past activations will affect the final logits. The cache contains those so we don't
        # need to recompute them. This is useful for generating text. As we have absolute positional
        # encodings, to implement this we have a `pos_offset` variable, defaulting to zero, which says
        # to offset which positional encodings are used (cached keys and values were calculated with
        # their own positional encodings).
        if past_kv_cache is None:
            pos_offset = 0
        else:
            batch_size, ctx_length = tokens.shape
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            assert num_heads_in_cache == self.cfg.n_heads
            assert d_head_in_cache == self.cfg.d_head
            # If we want to generate from the empty string, we'd pass in an empty cache, so we need to handle that case
            assert (
                cache_ctx_length == 0 or ctx_length == 1
            ), "Pass in one token at a time after loading cache"
            pos_offset = cache_ctx_length
        if self.cfg.use_hook_tokens:
            tokens = self.hook_tokens(tokens)
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        if self.cfg.custom_type == "embeddings":
            embed = self.hook_embed(self.cfg.embeddings)
            pos_embed = self.hook_pos_embed(torch.zeros(embed.shape))
            residual = embed
            shortformer_pos_embed = None
        elif self.cfg.custom_type == "token_only":
            pos_embed = self.hook_pos_embed(torch.zeros(embed.shape))
            residual = embed
            shortformer_pos_embed = None
        elif self.cfg.custom_type == "position_only":
            pos_offset = 0
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, left_attention_mask)
            )  # [batch, pos, d_model]
            embed = self.hook_embed(torch.zeros(embed.shape))
            residual = pos_embed  # [batch, pos, d_model]
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "standard":
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, left_attention_mask)
            )  # [batch, pos, d_model]
            residual = embed + pos_embed  # [batch, pos, d_model]
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "shortformer":
            # If we're using shortformer style attention, we don't add the positional embedding to the residual stream.
            # See HookedTransformerConfig for details
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, left_attention_mask)
            )  # [batch, pos, d_model]
            residual = embed
            shortformer_pos_embed = pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            # Rotary doesn't use positional embeddings, instead they're applied when dot producting keys and queries.
            # See HookedTransformerConfig for details
            residual = embed
            shortformer_pos_embed = None
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )

        if stop_at_layer is None:
            # We iterate through every block by default
            transformer_block_list = self.blocks
        else:
            # If we explicitly want to stop at a layer, we only iterate through the blocks up to that layer. Note that
            # this is exclusive, eg stop_at_layer==0 means to only run the embed, stop_at_layer==-1 means to run every
            # layer *apart* from the final one, etc.
            transformer_block_list = self.blocks[:stop_at_layer]  # type: ignore

        for i, block in enumerate(transformer_block_list):  # type: ignore
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            # If we're using multiple GPUs, we need to send the residual and shortformer_pos_embed to the correct GPU
            residual = residual.to(devices.get_device_for_block_index(i, self.cfg))
            if shortformer_pos_embed is not None:
                shortformer_pos_embed = shortformer_pos_embed.to(
                    devices.get_device_for_block_index(i, self.cfg)
                )

            residual = block(
                residual,
                past_kv_cache_entry=past_kv_cache[i]
                if past_kv_cache is not None
                else None,  # Cache contains a list of HookedTransformerKeyValueCache objects, one for each block
                shortformer_pos_embed=shortformer_pos_embed,
                left_attention_mask=left_attention_mask,
            )  # [batch, pos, d_model]

        if stop_at_layer is not None:
            # When we stop at an early layer, we end here rather than doing further computation
            return None

        if self.cfg.normalization_type is not None:
            residual = self.ln_final(residual)  # [batch, pos, d_model]
        if return_type is None:
            return None
        else:
            logits = self.unembed(residual)  # [batch, pos, d_vocab]
            if return_type == "logits":
                return logits
            else:
                loss = self.loss_fn(logits, tokens, per_token=loss_per_token)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return Output(logits, loss)
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None

  @classmethod
  def from_pretrained(
      cls,
      model_name: str,
      fold_ln=True,
      center_writing_weights=True,
      center_unembed=True,
      refactor_factored_attn_matrices=False,
      checkpoint_index=None,
      checkpoint_value=None,
      hf_model=None,
      device=None,
      n_devices=1,
      tokenizer=None,
      move_to_device=True,
      fold_value_biases=True,
      default_prepend_bos=True,
      default_padding_side="right",
      **from_pretrained_kwargs,
  ) -> "HookedTransformer":
      """Class method to load in a pretrained model weights to the HookedTransformer format and optionally to do some
      processing to make the model easier to interpret. Currently supports loading from most autoregressive
      HuggingFace models (GPT2, GPTNeo, GPTJ, OPT) and from a range of toy models and SoLU models trained by me (Neel Nanda).

      Also supports loading from a checkpoint for checkpointed models (currently, models trained by me (NeelNanda) and
      the stanford-crfm models). These can either be determined by the checkpoint index (the index of the checkpoint
      in the checkpoint list) or by the checkpoint value (the value of the checkpoint, eg 1000 for a checkpoint taken
      at step 1000 or after 1000 tokens. Each model has checkpoints labelled with exactly one of labels and steps).
      If neither is specified the final model is loaded. If both are specified, the checkpoint index is used.

      See load_and_process_state_dict for details on the processing (folding layer norm, centering the unembedding and
      centering the writing weights)

      Args:
          model_name (str): The model name - must be an element of OFFICIAL_MODEL_NAMES or an alias of one.
          fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the
              subsequent linear layer. This does not change the computation.
              Defaults to True.
          center_writing_weights (bool, optional): Whether to center weights
          writing to
              the residual stream (ie set mean to be zero). Due to LayerNorm
              this doesn't change the computation. Defaults to True.
          center_unembed (bool, optional): Whether to center W_U (ie set mean
          to be zero).
              Softmax is translation invariant so this doesn't affect log
              probs or loss, but does change logits. Defaults to True.
          refactor_factored_attn_matrices (bool, optional): Whether to convert the factored
              matrices (W_Q & W_K, and W_O & W_V) to be "even". Defaults to False
          checkpoint_index (int, optional): If loading from a checkpoint, the index of
              the checkpoint to load. Defaults to None.
          checkpoint_value (int, optional): If loading from a checkpoint, the value of
              the checkpoint to load, ie the step or token number (each model
              has checkpoints labelled with exactly one of these). Defaults to
              None.
          hf_model (AutoModelForCausalLM, optional): If you have already loaded in the
              HuggingFace model, you can pass it in here rather than needing
              to recreate the object. Defaults to None.
          device (str, optional): The device to load the model onto. By
              default will load to CUDA if available, else CPU.
          n_devices (int, optional): The number of devices to split the model
              across. Defaults to 1. If greater than 1, `device` must be cuda.
          tokenizer (*optional): The tokenizer to use for the model. If not
              provided, it is inferred from cfg.tokenizer_name or initialized to None.
              If None, then the model cannot be passed strings, and d_vocab must be explicitly set.
          move_to_device (bool, optional): Whether to move the model to the device specified in cfg.
              device. Must be true if `n_devices` in the config is greater than 1, since the model's layers
              will be split across multiple devices.
          default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
              methods of HookedTransformer process input text to tokenize (only when input is a string).
              Defaults to True - even for models not explicitly trained with this, heads often use the
              first position as a resting position and accordingly lose information from the first token,
              so this empirically seems to give better results. To change the default behavior to False, pass in
              default_prepend_bos=False. Note that you can also locally override the default behavior by passing
              in prepend_bos=True/False when you call a method that processes the input string.
          from_pretrained_kwargs (dict, optional): Any other optional argument passed to HuggingFace's
              from_pretrained (e.g. "cache_dir" or "torch_dtype"). Also passed to other HuggingFace
              functions when compatible. For some models or arguments it doesn't work, especially for
              models that are not internally loaded with HuggingFace's from_pretrained (e.g. SoLU models).
          default_padding_side (str, optional): Which side to pad on when tokenizing. Defaults to "right".
      """
      assert not (
          from_pretrained_kwargs.get("load_in_8bit", False)
          or from_pretrained_kwargs.get("load_in_4bit", False)
      ), "Quantization not supported"

      if from_pretrained_kwargs.get(
          "torch_dtype", None
      ) == torch.float16 and device in ["cpu", None]:
          logging.warning(
              "float16 models may not work on CPU. Consider using a GPU or bfloat16."
          )

      # Get the model name used in HuggingFace, rather than the alias.
      official_model_name = loading.get_official_model_name(model_name)

      # Load the config into an HookedTransformerConfig object. If loading from a
      # checkpoint, the config object will contain the information about the
      # checkpoint
      cfg = loading.get_pretrained_model_config(
          official_model_name,
          checkpoint_index=checkpoint_index,
          checkpoint_value=checkpoint_value,
          fold_ln=fold_ln,
          device=device,
          n_devices=n_devices,
          default_prepend_bos=default_prepend_bos,
          **from_pretrained_kwargs,
      )

      if cfg.positional_embedding_type == "shortformer":
          if fold_ln:
              logging.warning(
                  "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_"
                  "ln=False instead."
              )
              fold_ln = False
          if center_unembed:
              logging.warning(
                  "You tried to specify center_unembed=True for a shortformer model, but this can't be done! "
                  "Setting center_unembed=False instead."
              )
              center_unembed = False
          if center_writing_weights:
              logging.warning(
                  "You tried to specify center_writing_weights=True for a shortformer model, but this can't be done! "
                  "Setting center_writing_weights=False instead."
              )
              center_writing_weights = False

      # Get the state dict of the model (ie a mapping of parameter names to tensors), processed to match the
      # HookedTransformer parameter names.
      state_dict = loading.get_pretrained_state_dict(
          official_model_name, cfg, hf_model, **from_pretrained_kwargs
      )

      # Create the HookedTransformer object
      model = cls(
          cfg,
          tokenizer,
          move_to_device=False,
          default_padding_side=default_padding_side,
      )

      model.load_and_process_state_dict(
          state_dict,
          fold_ln=fold_ln,
          center_writing_weights=center_writing_weights,
          center_unembed=center_unembed,
          fold_value_biases=fold_value_biases,
          refactor_factored_attn_matrices=refactor_factored_attn_matrices,
      )

      if move_to_device:
          model.move_model_modules_to_device()

      print(f"Loaded pretrained model {model_name} into HookedTransformer")

      return model



def generate_tokens_arb_seq(model, seq_len, batch):
    '''
    Generates a sequence of repeated random tokens
    '''
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence[:, 1:-1] = first_sequence[:, 1:-1]
    token_sequence = torch.cat([prefix, first_sequence, second_sequence, first_sequence], dim=-1).to(device)

    alphabet = [' epsilon', ' zeta', ' eta', ' theta',
                ' iota', ' kappa', ' lambda', ' mu',
                ' nu', ' xi', ' omicron', ' pi', ' rho',
                ' sigma', ' tau', ' upsilon']

    randoms = alphabet[:seq_len-2]
    seq1 = [' alpha'] + randoms + [' beta']
    seq2 = [' gamma'] + randoms + [' delta']
    str_tokens = ['BOS'] + seq1 + seq2 + seq1

    ipos = -2

    return token_sequence, str_tokens, ipos


def generate_tokens_arb_seq2(model, seq_len, batch):
    '''
    Generates a sequence of repeated random tokens
    '''
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence[:, 1:-1] = first_sequence[:, 1:-1]
    token_sequence = torch.cat([prefix,
                                first_sequence, second_sequence,
                                first_sequence, second_sequence,
                                first_sequence, second_sequence,
                                first_sequence, second_sequence,
                                first_sequence], dim=-1).to(device)

    alphabet = [' epsilon', ' zeta', ' eta', ' theta',
                ' iota', ' kappa', ' lambda', ' mu',
                ' nu', ' xi', ' omicron', ' pi', ' rho',
                ' sigma', ' tau', ' upsilon']

    randoms = alphabet[:seq_len-2]
    seq1 = [' alpha'] + randoms + [' beta']
    seq2 = [' gamma'] + randoms + [' delta']
    str_tokens = ['BOS'] + seq1 + seq2 + seq1 + seq2 + seq1

    ipos = -2

    return token_sequence, str_tokens, ipos

def generate_tokens_arb_seq3(model, seq_len, batch):
    '''
    Generates a sequence of repeated random tokens
    '''
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    second_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    third_sequence = torch.randint(0, model.cfg.d_vocab, (batch, 10), dtype=torch.int64)
    second_sequence[:, 1:-1] = first_sequence[:, 1:-1]
    # token_sequence = torch.cat([prefix, third_sequence,
    #                             second_sequence, first_sequence, 
    #                             second_sequence, first_sequence, 
    #                             second_sequence, first_sequence, 
    #                             second_sequence, first_sequence,
    #                             third_sequence, 
    #                             first_sequence], dim=-1).to(device)
    token_sequence = torch.cat([prefix,
                                # third_sequence,
                                first_sequence,
                                second_sequence,
                                first_sequence], dim=-1).to(device)

    alphabet = [' epsilon', ' zeta', ' eta', ' theta',
                ' iota', ' kappa', ' lambda', ' mu',
                ' nu', ' xi', ' omicron', ' pi', ' rho',
                ' sigma', ' tau', ' upsilon']

    randoms = alphabet[:seq_len-2]
    seq1 = [' alpha'] + randoms + [' beta']
    seq2 = [' gamma'] + randoms + [' delta']
    str_tokens = ['BOS'] + seq1 + seq2 + seq1 + seq2 + seq1

    ipos = -2

    return token_sequence, str_tokens, ipos



def generate_tokens_abac_seq(model, seq_len, batch):
    '''
    Generates a sequence of repeated random tokens
    of the form: a, b, x, x, a, c, x, x, a

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    seq_len = 13 + 3
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    first_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    first_sequence[:, 6 + 3] = first_sequence[:, 1]
    first_sequence[:, 11 + 3] = first_sequence[:, 1]
    # We want a sequence of the form: a, b, x, x, a, c, x, x, a
    token_sequence = torch.cat([prefix, first_sequence], dim=-1).to(device)
    return token_sequence


def compute_lgts(logits, tokens, tokens_select, ipos=-2):
    # Compute probabilities for next tokens in repeated sequences
    lgts = {}
    output_tokens = tokens.index_select(1, tokens_select).T
    output_logits = logits[:, ipos, :]
    output_probs = logits[:, ipos, :].softmax(-1)

    lgts['logits1'] = output_logits[np.arange(0, output_logits.shape[0]), output_tokens[0]]
    lgts['logits2'] = output_logits[np.arange(0, output_logits.shape[0]), output_tokens[1]]
    lgts['probs1'] = output_probs[np.arange(0, output_probs.shape[0]), output_tokens[0]]
    lgts['probs2'] = output_probs[np.arange(0, output_probs.shape[0]), output_tokens[1]]

    lgts['mean_logit_diff'] = (lgts['logits1'] - lgts['logits2']).mean()
    lgts['p1mean'] = lgts['probs1'].mean()
    lgts['p2mean'] = lgts['probs2'].mean()
    lgts['pmean'] = output_probs.mean()

    maxes = output_probs.argmax(-1)
    lgts['nmax0'] = (maxes == output_tokens[0]).type(torch.float64).mean()
    lgts['nmax16'] = (maxes == 16).type(torch.float64).mean()

    return lgts


def generate_tokens_abac_seq(model, **kwargs):
    '''
    Generates a sequence of repeated random tokens
    of the form: a, b, x, x, a, c, x, x, a

    ab_pos : position of [A][B]
    ac_pos : position of [A][C]

    Outputs are:
        rep_tokens: [batch, seq_len]
    '''
    seq_len = kwargs['seq_len']
    batch = kwargs['batch']
    ab_pos = kwargs['ab_pos']
    ac_pos = kwargs['ac_pos']

    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long() # tensor([[1]])
    main_sequence = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    if kwargs['prefix']:
        main_sequence[:, ac_pos-1] = main_sequence[:, ab_pos-1]
        main_sequence[:, -1] = main_sequence[:, ab_pos-1]
        token_sequence = torch.cat([prefix, main_sequence], dim=-1).to(device)
    else:
        main_sequence[:, ac_pos] = main_sequence[:, ab_pos]
        main_sequence[:, -1] = main_sequence[:, ab_pos]
        token_sequence = main_sequence        

    str_tokens = ['Random'] * seq_len

    if kwargs['prefix']:
        str_tokens[ab_pos-1] = 'A'
        str_tokens[ac_pos-1] = 'A'
        str_tokens[-1] = 'A'
        str_tokens[ab_pos] = 'B'
        str_tokens[ac_pos] = 'C'
        str_tokens = ['BOS'] + str_tokens
    else:
        str_tokens[ab_pos] = 'A'
        str_tokens[ac_pos] = 'A'
        str_tokens[-1] = 'A'
        str_tokens[ab_pos + 1] = 'B'
        str_tokens[ac_pos + 1] = 'C'

    return token_sequence, str_tokens

def genseq(pattern, model, **kwargs):
    pattern_to_fun = {
        'abac': generate_tokens_abac_seq
    }
    return pattern_to_fun[pattern](model, **kwargs)


def run_sequence(model, seq_len, head_ablation_hook, nbatch=500):
  # torch.manual_seed(130)
  rep_tokens, str_tokens, ipos = generate_tokens_arb_seq(model, seq_len, nbatch)

  # Run with hooks
  ilayer = 0

  pattern_store = torch.zeros((model.cfg.n_layers,
                               model.cfg.n_heads, len(rep_tokens[0]), len(rep_tokens[0])),
                               device=model.cfg.device)

  def pattern_hook(pattern, hook):
      avg_pattern = einops.reduce(pattern, "batch head_index p1 p2 -> head_index p1 p2", "mean")
      pattern_store[hook.layer()] = avg_pattern

  # We make a boolean filter on activation names, that's true only on attention pattern names
  pattern_hook_names_filter = lambda name: name.endswith("pattern")


  logits = model.run_with_hooks(
      rep_tokens,
      return_type='logits', # For efficiency, we don't need to calculate the logits
      fwd_hooks=[(utils.get_act_name("pattern", ilayer), head_ablation_hook),
                (pattern_hook_names_filter, pattern_hook)]
  )

  lgts = {}
  output_tokens = rep_tokens.index_select(1, torch.tensor([seq_len, seq_len*2])).T
  output_logits = logits[:, ipos, :]
  output_probs = logits[:, ipos, :].softmax(-1)

  lgts['logits1'] = output_logits[np.arange(0, output_logits.shape[0]), output_tokens[0]]
  lgts['logits2'] = output_logits[np.arange(0, output_logits.shape[0]), output_tokens[1]]
  lgts['probs1'] = output_probs[np.arange(0, output_probs.shape[0]), output_tokens[0]]
  lgts['probs2'] = output_probs[np.arange(0, output_probs.shape[0]), output_tokens[1]]

  lgts['mean_logit_diff'] = (lgts['logits1'] - lgts['logits2']).mean()
  lgts['p1mean'] = lgts['probs1'].mean()
  lgts['p2mean'] = lgts['probs2'].mean()
  lgts['pmean'] = output_probs.mean()

  maxes = output_probs.argmax(-1)
  lgts['nmax0'] = (maxes == output_tokens[0]).type(torch.float64).mean()
  lgts['nmax16'] = (maxes == 16).type(torch.float64).mean()

  return rep_tokens, str_tokens, pattern_store, logits, lgts

