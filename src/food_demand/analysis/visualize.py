
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def create_visualizations(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Generate charts from the EDA report files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Historical weekly demand
    weekly = pd.read_csv(input_dir / "weekly_demand.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly["week"], weekly["num_orders"])
    ax.set(
        title="Historical Weekly Food Demand",
        xlabel="Week",
        ylabel="Total Orders",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "weekly_demand.png", dpi=160)
    plt.close(fig)

    # 2. Top five food categories
    categories = pd.read_csv(
        input_dir / "category_demand.csv"
    ).head(5)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        categories["category"],
        categories["num_orders"],
    )
    ax.invert_yaxis()
    ax.set(
        title="Top 5 Food Categories by Order Volume",
        xlabel="Total Orders",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "top_categories.png", dpi=160)
    plt.close(fig)

    # 3. Promotion comparison
    promo = pd.read_csv(
        input_dir / "promotion_comparison.csv"
    )

    labels = {
        (0, 0): "Neither",
        (0, 1): "Homepage",
        (1, 0): "Email",
        (1, 1): "Both",
    }

    promo["promotion_type"] = [
        labels[(int(email), int(homepage))]
        for email, homepage in zip(
            promo["emailer_for_promotion"],
            promo["homepage_featured"],
        )
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        promo["promotion_type"],
        promo["average_orders"],
    )
    ax.set(
        title="Average Orders by Promotion Type",
        ylabel="Average Orders per Observation",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "promotion_comparison.png", dpi=160)
    plt.close(fig)

    print("Three visualizations generated successfully!")


if __name__ == "__main__":
    create_visualizations(
        Path("data/processed/eda"),
        Path("data/processed/visualizations"),
    )
