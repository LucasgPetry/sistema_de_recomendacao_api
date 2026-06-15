from pathlib import Path

import pandas as pd

INPUT = Path("data/raw/ratings.csv")
OUTPUT = Path("data/processed/ratings_sample.csv")

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_SIZE = 100_000
CHUNK_SIZE = 100_000

sample_parts = []
current_size = 0

print("Gerando amostra...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT,
        chunksize=CHUNK_SIZE
    )
):

    sampled_chunk = chunk.sample(
        frac=0.01,
        random_state=42
    )

    sample_parts.append(sampled_chunk)

    current_size += len(sampled_chunk)

    print(
        f"Chunk {chunk_number} | "
        f"Amostra acumulada: "
        f"{current_size:,}"
    )

    if current_size >= TARGET_SIZE:
        break

sample = pd.concat(
    sample_parts,
    ignore_index=True
)

sample = sample.head(TARGET_SIZE)

sample.to_csv(
    OUTPUT,
    index=False
)

print(
    f"\nArquivo salvo em:\n{OUTPUT}"
)

print(
    f"Quantidade final:\n{len(sample):,}"
)