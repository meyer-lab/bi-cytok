import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ..BindingMod import polyc
from ..imports import importCITE
from scipy.optimize import minimize


def makeFigure():
    df = importCITE()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    # Plot IL2Ra distribution
    sns.boxplot(x="CellType1", y="CD25", data=df, ax=axes[0])
    axes[0].set_title("IL2Ra (CD25) Distribution")
    axes[0].set_xlabel("Cell Type")
    axes[0].set_ylabel("IL2Ra (CD25)")

    # Plot IL2RB distribution
    sns.boxplot(x="CellType1", y="CD122", data=df, ax=axes[1])
    axes[1].set_title("IL2RB (CD122) Distribution")
    axes[1].set_xlabel("Cell Type")
    axes[1].set_ylabel("IL2RB (CD122)")

    def find_best_markers(cell_type, df):
        filtered_df = df[df["CellType1"] == cell_type]
        mean_expression = filtered_df.select_dtypes(include='number').mean()
        mean_expression = mean_expression.drop(["CD25", "CD122"])
        rest_of_cells = df[df["CellType1"] != cell_type]
        mean_expression_rest = rest_of_cells.select_dtypes(include='number').mean()
        fold_change = mean_expression / mean_expression_rest
        sorted_markers = fold_change.sort_values(ascending=False)
        top_markers = sorted_markers.head(3)
        return top_markers.index.tolist()

    cell_type = "NK"
    top_markers = find_best_markers(cell_type, df)
    print(f"The top markers for {cell_type} are: {top_markers}")

    sample_df = df.sample(n=1000, random_state=51)
    sample_df = sample_df.reset_index(drop=True)

    nk_marker = []
    nk_cd122 = []
    non_nk_marker = []
    non_nk_cd122 = []

    # Filter NK cells
    nk_df = sample_df[sample_df["CellType1"] == "NK"]
    nk_marker = nk_df["CD335"].tolist()
    nk_cd122 = nk_df["CD122"].tolist()

    # Filter non-NK cells
    non_nk_df = sample_df[sample_df["CellType1"] != "NK"]
    non_nk_marker = non_nk_df["CD335"].tolist()
    non_nk_cd122 = non_nk_df["CD122"].tolist()

    def optimization_function(Kav):
        Kav_scalar = Kav[0]
        Kav_matrix = np.array([[1e8, 1e2], [1e2, np.power(10, Kav_scalar)]])
        NKIL2_bound = 0
        non_NKIL2_bound = 0

        L0 = 1e-9
        KxStar = 1e-12
        cplx = np.array(
            [[1, 1]]
        )  # Changed - before this was saying there are two different types of ligand, now its a single ligand with two components

        # Got rid of initial loop

        for i, _ in enumerate(nk_marker):
            NKbinding, _ = polyc(
                L0,
                KxStar,
                np.array([nk_cd122[i] * 100, nk_marker[i] * 100]),
                cplx,
                Kav_matrix,
            )
            NKIL2_bound += NKbinding[0][0]
        for i, _ in enumerate(non_nk_marker):
            non_NKbinding, _ = polyc(
                L0,
                KxStar,
                np.array([non_nk_cd122[i] * 100, non_nk_marker[i] * 100]),
                cplx,
                Kav_matrix,
            )
            non_NKIL2_bound += non_NKbinding[0][0]
        return non_NKIL2_bound / NKIL2_bound

    initial_guess = [7]
    result = minimize(optimization_function, initial_guess, bounds=[(5, 10)])
    optimized_Kav = result.x[0]
    print(f"Optimized Kav: {optimized_Kav}")

    return fig
