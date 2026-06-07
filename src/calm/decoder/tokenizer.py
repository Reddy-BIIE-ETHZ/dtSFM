"""Tokenizer for CALM-Decoder.

Provides tokenizer for decoding side only, as the encoding side takes continuous embeddings.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from transformers import BatchEncoding, T5TokenizerFast


class CALMTokenizer(T5TokenizerFast):
    """Tokenizer for CALM-Decoder.

    A specialized tokenizer that extends T5TokenizerFast for handling protein sequences
    in the CALM (Continuous Antibody Language Model) decoder. This tokenizer supports
    special tokens for antigens (<ag>) and antibodies (<ab>) and provides methods for
    encoding protein sequences with appropriate special tokens.

    Parameters
    ----------
    vocab_file : str
        Path to the vocabulary file for the tokenizer.
    ag_token : str, optional
        Special token for antigens.
    ab_token : str, optional
        Special token for antibodies.
    **kwcfg
        Additional keyword arguments passed to the parent T5TokenizerFast.

    Attributes
    ----------
    ag_token : str
        The antigen special token.
    ab_token : str
        The antibody special token.
    """

    def __init__(
        self,
        vocab_file: str,
        ag_token: str,
        ab_token: str,
        **kwcfg: Any,
    ):
        """Initialize the CALMTokenizer.

        Parameters
        ----------
        vocab_file : str
            Path to the SentencePiece vocabulary (.model) file.
        ag_token : str
            Special token string for antigens (e.g. ``"<ag>"``)
        ab_token : str
            Special token string for antibodies (e.g. ``"<ab>"``)
        **kwcfg
            Additional keyword arguments forwarded to
            ``T5TokenizerFast.__init__`` (e.g. ``extra_ids``, ``legacy``).
        """
        kwcfg.setdefault("legacy", False)
        kwcfg.setdefault("extra_ids", 0)

        super().__init__(vocab_file=vocab_file, **kwcfg)  # type: ignore[no-untyped-call]
        self.ag_token = ag_token
        self.ab_token = ab_token

    def tokenize_batch(
        self,
        seq: str | list[str],
        add_special_tokens: bool = True,
        padding: bool | str = "longest",
        truncation: bool = False,
        return_tensors: str | None = "pt",
        **kwcfg: Any,
    ) -> BatchEncoding:
        r"""Tokenize a sequence or batch of sequences.

        Runs the full HuggingFace tokenization pipeline (IDs + masks).
        Since T5 does not use a BOS token, setting ``add_special_tokens=True``
        only appends an EOS token (``</s>``).

        Parameters
        ----------
        seq : str | list[str]
            A single sequence string or a list of sequence strings.
        add_special_tokens : bool, optional
            Whether to add EOS tokens, by default True.
        padding : bool | str, optional
            Padding strategy passed to the HuggingFace tokenizer, by default
            ``"longest"``.
        truncation : bool, optional
            Whether to truncate sequences exceeding ``max_length``, by default
            False.
        return_tensors : str | None, optional
            Format of the returned tensors (e.g. ``"pt"`` for PyTorch), by
            default ``"pt"``.
        **kwcfg
            Additional keyword arguments forwarded to the parent
            ``T5TokenizerFast.__call__``.

        Returns
        -------
        BatchEncoding
            A HuggingFace ``BatchEncoding`` containing ``input_ids`` and
            ``attention_mask`` (and any other requested fields).
        """
        texts = [seq] if isinstance(seq, str) else list(seq)

        return BatchEncoding(
            super().__call__(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                **kwcfg,
            )
        )

    def encode_labels(
        self,
        seqs: list[str],
        prepend_token: str | None = None,
        add_eos: bool = True,
    ) -> list[list[int]]:
        """Encode sequences into label ID lists.

        Parameters
        ----------
        seqs : list[str]
            List of sequence strings to encode.
        prepend_token : str | None, optional
            Token to prepend to each sequence (e.g. ``"<ag>"`` or ``"<ab>"``),
            by default None.
        add_eos : bool, optional
            Whether to append the EOS token ID to each sequence, by default True.

        Returns
        -------
        list[list[int]]
            List of token ID lists, one per input sequence.
        """
        out = []
        for s in seqs:
            if prepend_token:
                s = prepend_token + s

            ids = super().__call__(s, add_special_tokens=False, return_tensors=None)[
                "input_ids"
            ]
            if add_eos:
                ids.append(self.eos_token_id)
            out.append(ids)
        return out


def _stack_reduce_embeddings(
    embed: Tensor, mask: Tensor, reduce_batch_padding: bool = True
) -> tuple[Tensor, Tensor]:
    """Stack embeddings and masks, optionally removing excess padding.

    Parameters
    ----------
    embed : Tensor
        Input embeddings to stack into a batch tensor.
    mask : Tensor
        Attention masks corresponding to the embeddings.
    reduce_batch_padding : bool, optional
        Whether to trim the batch to the maximum valid length to reduce
        unnecessary padding, by default True.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing:
        - embed: Stacked embeddings tensor of shape [B, L, D] or [B, L_valid, D]
        - mask: Stacked attention mask tensor of shape [B, L] or [B, L_valid]
    """
    embed = torch.stack(list(embed))  # [B, L, D]
    mask = torch.stack(list(mask))  # [B, L]

    if reduce_batch_padding:
        # Remove extra padding in the batch (if any)
        valid_lengths = int(mask.sum(dim=1).max().item())  # max valid length in batch
        embed = embed[:, :valid_lengths, :]  # [B, L_valid, D]
        mask = mask[:, :valid_lengths]  # [B, L_valid]

    return embed, mask


@dataclass
class CALMCollator:
    """Collate function that tokenizes and pads batch items with CALMTokenizer.

    - Adds EOS only when add_special_tokens=True (HF does that for both src/tgt here)
    - Replaces PADs in labels with -100 for T5-style loss
    - Optionally prepend your task tokens (<ag>, <ab>) to inputs/targets
    """

    tokenizer: CALMTokenizer
    max_tgt_len_ag: int | None = None
    max_tgt_len_ab: int | None = None
    pad_value: float = 0.0
    include_text_in_batch: bool = False

    def __call__(
        self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor, str, str]]
    ) -> dict[str, Any]:
        """Collate batch items for bidirectional antigen-antibody generation.

        Parameters
        ----------
        batch : List[Tuple[Tensor, Tensor, Tensor, Tensor, str, str]]
            A list of batch items where each item contains:
            - ag_embed: Antigen embedding tensor
            - ab_embed: Antibody embedding tensor
            - ag_mask: Attention mask for antigen embeddings
            - ab_mask: Attention mask for antibody embeddings
            - ag_seq: Antigen sequence string
            - ab_seq: Antibody sequence string

        Returns
        -------
        Dict[str, Any]
            A dictionary containing bidirectional training data with keys:
            - "ag2ab": Dict with inputs_embeds, attention_mask, labels for antigen→antibody
            - "ab2ag": Dict with inputs_embeds, attention_mask, labels for antibody→antigen
            - "ag_seqs": List of antigen sequences (if include_text_in_batch=True)
            - "ab_seqs": List of antibody sequences (if include_text_in_batch=True)
        """
        # ag_embs: Tuple[Tensor], ab_embs: Tuple[Tensor], ag_texts: Tuple[str], ab_texts: Tuple[str]
        ag_embed, ab_embed, ag_mask, ab_mask, ag_seq, ab_seq = zip(*batch, strict=False)

        # --- AG -> AB path ---
        ag_embed_src, ag_embed_mask = _stack_reduce_embeddings(ag_embed, ag_mask)  # type: ignore[arg-type]
        ab_seq_tgt = list(ab_seq)
        ab_seq_tgt_tokenized = self.tokenizer.tokenize_batch(
            seq=ab_seq_tgt,
            add_special_tokens=True,
            padding=True,
            truncation=self.max_tgt_len_ab is not None,
            max_length=self.max_tgt_len_ab,
            return_tensors="pt",
            return_attention_mask=False,
        )
        ab_seq_label = ab_seq_tgt_tokenized["input_ids"]
        ab_seq_label[ab_seq_label == self.tokenizer.pad_token_id] = -100

        # --- AB -> AG path ---
        ab_embed_src, ab_embed_mask = _stack_reduce_embeddings(ab_embed, ab_mask)  # type: ignore[arg-type]
        ag_seq_tgt = list(ag_seq)
        ag_seq_tgt_tokenized = self.tokenizer.tokenize_batch(
            seq=ag_seq_tgt,
            add_special_tokens=True,
            padding=True,
            truncation=self.max_tgt_len_ag is not None,
            max_length=self.max_tgt_len_ag,
            return_tensors="pt",
            return_attention_mask=False,
        )
        ag_seq_label = ag_seq_tgt_tokenized["input_ids"]
        ag_seq_label[ag_seq_label == self.tokenizer.pad_token_id] = -100

        out: dict[str, Any] = {
            "ag2ab": {
                "inputs_embeds": ag_embed_src,  # [B, L_ag_max, D]
                "attention_mask": ag_embed_mask,  # [B, L_ag_max]
                "labels": ab_seq_label,  # [B, T_ab]
            },
            "ab2ag": {
                "inputs_embeds": ab_embed_src,  # [B, L_ab_max, D]
                "attention_mask": ab_embed_mask,  # [B, L_ab_max]
                "labels": ag_seq_label,  # [B, T_ag]
            },
        }
        if self.include_text_in_batch:
            out["ag_seqs"] = list(ag_seq)
            out["ab_seqs"] = list(ab_seq)

        return out
