import pandas as pd

print("Carregando ratings...")

ratings = pd.read_csv(
    "data/processed/ratings_harmonized.csv"
)

user_activity = (
    ratings["user_id"]
    .value_counts()
)

print(
    f"\nUsuários únicos: "
    f"{len(user_activity):,}"
)

print("\n=== ESTATÍSTICAS ===")
print(
    user_activity.describe()
)

bins = {
    "1 avaliação": (
        user_activity == 1
    ).sum(),

    "2-5 avaliações": (
        (user_activity >= 2)
        &
        (user_activity <= 5)
    ).sum(),

    "6-10 avaliações": (
        (user_activity >= 6)
        &
        (user_activity <= 10)
    ).sum(),

    "11-20 avaliações": (
        (user_activity >= 11)
        &
        (user_activity <= 20)
    ).sum(),

    "21+ avaliações": (
        user_activity >= 21
    ).sum(),
}

print("\n=== DISTRIBUIÇÃO ===")

total_users = len(
    user_activity
)

for label, count in bins.items():

    percentage = (
        count
        /
        total_users
        *
        100
    )

    print(
        f"{label:<18}"
        f"{count:>6}"
        f" usuários "
        f"({percentage:.2f}%)"
    )