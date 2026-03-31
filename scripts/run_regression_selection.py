from pathlib import Path

from src.data.loaders import load_master_dataframe
from src.models.regression_selection import evaluate_models, select_xy_from_master


def main() -> None:
    df = load_master_dataframe(sheet_name="experimental")
    split = select_xy_from_master(df)

    results_df = evaluate_models(split.X, split.Y)

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "regression_model_selection_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nTop models:")
    print(results_df.head(10).to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()