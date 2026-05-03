from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "bilstm_attention_dataset.jsonl"
MODEL_PATH = BASE_DIR / "spell_model.pt"
SPECIAL_TOKENS = {"<PAD>", "<SOS>", "<EOS>"}
MAX_EDIT_DISTANCE = 2


@dataclass
class LookupTables:
    exact_corrections: dict[str, str]
    noisy_forms: list[str]
    targets: set[str]


class SpellSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.decoder = nn.LSTM(
            embedding_dim + hidden_size * 2,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention_query = nn.Linear(hidden_size, hidden_size * 2)
        self.output = nn.Linear(hidden_size, vocab_size)

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.emb(source)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        hidden = self._merge_bidir_state(hidden)
        cell = self._merge_bidir_state(cell)
        return encoder_outputs, (hidden, cell)

    def _merge_bidir_state(self, state: torch.Tensor) -> torch.Tensor:
        state = state.view(self.num_layers, 2, state.size(1), self.hidden_size)
        return state.sum(dim=1)

    def decode_step(
        self,
        decoder_input: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.emb(decoder_input)
        query = self.attention_query(state[0][-1])
        scores = torch.bmm(encoder_outputs, query.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~source_mask, -1e9)
        attention = torch.softmax(scores, dim=1)
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs)
        decoder_features = torch.cat([embedded, context], dim=2)
        decoder_output, state = self.decoder(decoder_features, state)
        logits = self.output(decoder_output)
        return logits, state

    def forward(self, source: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:
        encoder_outputs, state = self.encode(source)
        source_mask = source.ne(0)
        step_logits: list[torch.Tensor] = []
        for step in range(decoder_inputs.size(1)):
            logits, state = self.decode_step(
                decoder_inputs[:, step : step + 1],
                state,
                encoder_outputs,
                source_mask,
            )
            step_logits.append(logits)
        return torch.cat(step_logits, dim=1)


def load_jsonl_dataset(path: Path = DATASET_PATH) -> list[dict[str, list[str]]]:
    rows: list[dict[str, list[str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_vocab(data: list[dict[str, list[str]]]) -> tuple[dict[str, int], dict[int, str]]:
    chars = set()
    for row in data:
        chars.update(row["input"])
        chars.update(row["target"])

    chars.update(SPECIAL_TOKENS)
    char2idx = {char: index for index, char in enumerate(sorted(chars))}
    idx2char = {index: char for char, index in char2idx.items()}
    return char2idx, idx2char


def build_lookup_tables(data: list[dict[str, list[str]]]) -> LookupTables:
    noisy_to_target: dict[str, Counter[str]] = defaultdict(Counter)
    target_forms: set[str] = set()

    for row in data:
        noisy = "".join(row["input"][1:-1])
        clean = "".join(row["target"][1:-1])
        noisy_to_target[noisy][clean] += 1
        target_forms.add(clean)

    exact_corrections = {
        noisy: counts.most_common(1)[0][0]
        for noisy, counts in noisy_to_target.items()
    }
    return LookupTables(
        exact_corrections=exact_corrections,
        noisy_forms=list(exact_corrections.keys()),
        targets=target_forms,
    )


class SpellDataset(Dataset):
    def __init__(self, data: list[dict[str, list[str]]], char2idx: dict[str, int]) -> None:
        self.data = data
        self.char2idx = char2idx

    def encode(self, seq: list[str]) -> list[int]:
        return [self.char2idx[token] for token in seq]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.encode(self.data[index]["input"])
        target = self.encode(self.data[index]["target"])
        return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def build_collate_fn(pad_idx: int):
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*batch)
        max_source = max(len(x) for x in xs)
        max_target = max(len(y) for y in ys)

        def pad(seq: torch.Tensor, max_len: int) -> torch.Tensor:
            if len(seq) == max_len:
                return seq
            padding = torch.full((max_len - len(seq),), pad_idx, dtype=torch.long)
            return torch.cat([seq, padding])

        batch_x = torch.stack([pad(x, max_source) for x in xs])
        batch_y = torch.stack([pad(y, max_target) for y in ys])
        return batch_x, batch_y

    return collate


def is_word_token(token: str) -> bool:
    if not token:
        return False

    has_text = False
    for char in token:
        if char in {"'", "-"}:
            continue

        category = unicodedata.category(char)
        if category[0] in {"L", "M", "N"}:
            has_text = True
            continue

        return False

    return has_text


def levenshtein_distance(left: str, right: str, max_distance: int | None = None) -> int:
    if left == right:
        return 0

    if not left:
        return len(right)
    if not right:
        return len(left)

    if max_distance is not None and abs(len(left) - len(right)) > max_distance:
        return max_distance + 1

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            value = min(insert_cost, delete_cost, replace_cost)
            current.append(value)
            row_min = min(row_min, value)

        if max_distance is not None and row_min > max_distance:
            return max_distance + 1

        previous = current

    return previous[-1]


def nearest_lookup_correction(word: str, lookup_tables: LookupTables) -> str | None:
    best_distance: int | None = None
    best_candidate: str | None = None

    for candidate in lookup_tables.noisy_forms:
        distance = levenshtein_distance(word, candidate, MAX_EDIT_DISTANCE)
        if distance > MAX_EDIT_DISTANCE:
            continue

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_candidate = candidate
            if distance == 1:
                break

    if best_candidate is None:
        return None

    return lookup_tables.exact_corrections[best_candidate]


def load_spell_model(
    checkpoint_path: Path = MODEL_PATH,
    dataset_path: Path = DATASET_PATH,
    device: str | torch.device = "cpu",
) -> tuple[SpellSeq2Seq, dict[str, int], dict[int, str], LookupTables]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    config = checkpoint.get("config", {})
    model = SpellSeq2Seq(
        vocab_size=len(char2idx),
        embedding_dim=config.get("embedding_dim", 128),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lookup_tables = checkpoint.get("lookup_tables")
    if lookup_tables is None:
        data = load_jsonl_dataset(dataset_path)
        lookup_tables = build_lookup_tables(data)
    else:
        lookup_tables = LookupTables(
            exact_corrections=lookup_tables["exact_corrections"],
            noisy_forms=lookup_tables["noisy_forms"],
            targets=set(lookup_tables["targets"]),
        )

    return model, char2idx, idx2char, lookup_tables


def predict_word(
    word: str,
    model: SpellSeq2Seq,
    char2idx: dict[str, int],
    idx2char: dict[int, str],
    lookup_tables: LookupTables,
    max_output_length: int | None = None,
) -> str:
    if not word:
        return word

    exact = lookup_tables.exact_corrections.get(word)
    if exact is not None:
        return exact

    nearest = nearest_lookup_correction(word, lookup_tables)
    if nearest is not None:
        return nearest

    if any(char not in char2idx for char in word):
        return word

    source_tokens = ["<SOS>", *word, "<EOS>"]
    source = torch.tensor(
        [[char2idx[token] for token in source_tokens]],
        dtype=torch.long,
    )
    source_mask = source.ne(char2idx["<PAD>"])
    encoder_outputs, state = model.encode(source)

    generated = [char2idx["<SOS>"]]
    max_steps = max_output_length or max(len(word) + 3, 8)

    with torch.no_grad():
        for _ in range(max_steps):
            decoder_input = torch.tensor([[generated[-1]]], dtype=torch.long)
            logits, state = model.decode_step(decoder_input, state, encoder_outputs, source_mask)
            next_token = int(logits[:, -1, :].argmax(dim=-1).item())
            if idx2char[next_token] == "<EOS>":
                break
            generated.append(next_token)

    corrected = "".join(
        idx2char[token_id]
        for token_id in generated[1:]
        if idx2char[token_id] not in SPECIAL_TOKENS
    )
    return corrected or word


def correct_sentence(
    sentence: str,
    model: SpellSeq2Seq,
    char2idx: dict[str, int],
    idx2char: dict[int, str],
    lookup_tables: LookupTables,
) -> tuple[list[tuple[str, str]], str]:
    tokens = re.split(r"(\s+)", sentence)
    corrected_tokens: list[str] = []
    mistakes: list[tuple[str, str]] = []

    for token in tokens:
        if is_word_token(token):
            prediction = predict_word(token, model, char2idx, idx2char, lookup_tables)
            if prediction != token:
                mistakes.append((token, prediction))
            corrected_tokens.append(prediction)
        else:
            corrected_tokens.append(token)

    return mistakes, "".join(corrected_tokens)


def split_dataset(
    data: list[dict[str, list[str]]],
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, list[str]]], list[dict[str, list[str]]]]:
    if len(data) < 2:
        return data, data

    items = list(data)
    random.Random(seed).shuffle(items)
    validation_size = max(1, int(len(items) * validation_fraction))
    validation_data = items[:validation_size]
    training_data = items[validation_size:]
    return training_data, validation_data
