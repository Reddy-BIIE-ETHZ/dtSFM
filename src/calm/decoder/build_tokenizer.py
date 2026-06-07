"""Build SentencePiece tokenizer for CALM decoder."""

import os

import pandas as pd
import sentencepiece as spm
from omegaconf import DictConfig


def build(cfg: DictConfig, db_file: str, db_col: str, vocab_file: str) -> None:
    """Build a SentencePiece tokenizer.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing tokenizer settings.
    db_file : str
        The path to the input database file.
    db_col : str
        The name of the column containing sequences.
    vocab_file : str
        The path to the output vocabulary file.
    """
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)

    corpus_file = vocab_file.replace(".model", ".txt")
    if not os.path.exists(corpus_file):
        prepare_corpus(db_file, corpus_file, db_col)

    prepare_spm(cfg, corpus_file, out_prefix=vocab_file.replace(".model", ""))


def prepare_corpus(infile: str, outfile: str, col: str) -> None:
    """Prepare corpus file for tokenizer training.

    Parameters
    ----------
    infile : str
        The path to the input corpus file.
    outfile : str
        The path to the output corpus file.
    col : str
        The column name containing sequences.
    """
    df = pd.read_csv(infile)
    seq_list = list(set(df[col].dropna().tolist()))
    with open(outfile, "w") as fout:
        for seq in seq_list:
            fout.write(seq + "\n")
    print(f"Tokenizer Corpus: {outfile}: {len(seq_list)} sequences")


def check_token_coverage(corpus_file: str, token_provided: list[str]) -> None:
    """Check if all characters in the corpus are covered by the provided tokens.

    Parameters
    ----------
    corpus_file : str
        The path to the corpus file.
    token_provided : list of str
        The list of tokens that should cover the corpus characters.
    """
    with open(corpus_file) as f:
        corpus = f.read().replace("\n", "")

    unique_chars = set(corpus)
    print(f"Unique characters in corpus: {unique_chars}")
    provided_set = set(token_provided)
    print(f"Provided tokens: {provided_set}")

    uncovered_chars = unique_chars - provided_set
    if uncovered_chars:
        raise ValueError(
            f"Error: The following characters are not covered by the provided tokens: {uncovered_chars}"
        )
    else:
        print("All characters in the corpus are covered by the provided tokens.")


def prepare_spm(cfg: DictConfig, corpus_file: str, out_prefix: str) -> None:
    """Prepare SentencePiece model training.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing tokenizer settings.
    corpus_file : str
        The path to the input corpus file.
    out_prefix : str
        The prefix for the output model files.
    """
    aa_list = list(cfg.aa_list)
    user_symbols = cfg.symbol_list
    control_token_ids = {
        "pad": cfg.pad_id,
        "eos": cfg.eos_id,
        "unk": cfg.unk_id,
        "bos": cfg.bos_id,
    }

    num_control_tokens = sum(i >= 0 for i in control_token_ids.values())
    vocab_size = num_control_tokens + len(aa_list) + len(user_symbols)

    check_token_coverage(corpus_file, token_provided=aa_list + user_symbols)

    spm.SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=out_prefix,
        model_type="unigram",
        vocab_size=vocab_size,
        pad_id=cfg.pad_id,
        pad_piece=cfg.pad_token,
        eos_id=cfg.eos_id,
        eos_piece=cfg.eos_token,
        unk_id=cfg.unk_id,
        unk_piece=cfg.unk_token,
        bos_id=cfg.bos_id,
        user_defined_symbols=",".join(user_symbols),
        character_coverage=1.0,
        input_sentence_size=1000,
        shuffle_input_sentence=True,
        add_dummy_prefix=False,  # key change
        split_by_whitespace=False,  # avoid whitespace-derived '▁' tokens
        remove_extra_whitespaces=False,
    )
