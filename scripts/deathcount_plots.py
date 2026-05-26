import os
import tempfile
from pathlib import Path

TMP_DIR = tempfile.mkdtemp(prefix="plot_cache_")
os.environ.setdefault("MPLCONFIGDIR", TMP_DIR)
os.environ.setdefault("XDG_CACHE_HOME", TMP_DIR)

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "../data"
FIGURES_DIR = BASE_DIR / "../figures"
TOP_N = 5

DATASETS = [
    {
        "input": DATA_DIR / "county_drug_death.csv",
        "output": FIGURES_DIR / "county_drug_death_cumulative_top5.png",
        "title": "Cumulative Drug Death Counts by County",
        "ylabel": "Cumulative deaths",
    },
    {
        "input": DATA_DIR / "pop_normalized_county_drug_death.csv",
        "output": FIGURES_DIR / "pop_normalized_county_drug_death_cumulative_top5.png",
        "title": "Population-Normalized Cumulative Drug Death Counts by County",
        "ylabel": "Cumulative deaths per resident",
    },
]


def load_cumulative_counts(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).rename(columns={"Unnamed: 0": "county"})
    df = df.loc[~df["county"].astype(str).str.fullmatch("Total", case=False, na=False)].copy()
    monthly_columns = [column for column in df.columns if column != "county"]
    long_df = df.melt(
        id_vars="county",
        value_vars=monthly_columns,
        var_name="month",
        value_name="deaths",
    )
    long_df["date"] = pd.to_datetime(long_df["month"], format="%Y_%m")
    wide_df = (
        long_df.pivot(index="date", columns="county", values="deaths")
        .sort_index()
        .fillna(0)
    )
    return wide_df.cumsum()


def top_n_masked(cumulative_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=cumulative_df.index, columns=cumulative_df.columns)
    for date, row in cumulative_df.iterrows():
        top_counties = row.nlargest(top_n).index
        mask.loc[date, top_counties] = True
    return cumulative_df.where(mask)


def plot_top_n(cumulative_df: pd.DataFrame, output_path: Path, title: str, ylabel: str) -> None:
    filtered_df = top_n_masked(cumulative_df, TOP_N)
    visible_counties = filtered_df.notna().any().loc[lambda series: series].index
    plot_order = cumulative_df.loc[:, visible_counties].iloc[-1].sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 8))
    for county in plot_order:
        ax.plot(
            filtered_df.index,
            filtered_df[county],
            linewidth=2.0,
            alpha=0.9,
            label=county,
        )

    ax.set_title(f"{title}\nTop {TOP_N} counties shown at each month")
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(title="Counties", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for dataset in DATASETS:
        cumulative_df = load_cumulative_counts(dataset["input"])
        plot_top_n(
            cumulative_df=cumulative_df,
            output_path=dataset["output"],
            title=dataset["title"],
            ylabel=dataset["ylabel"],
        )
        print(f"Saved {dataset['output']}")


if __name__ == "__main__":
    main()
