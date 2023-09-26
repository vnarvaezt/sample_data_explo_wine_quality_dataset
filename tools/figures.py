import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns


def kdeplot_custom(df, target, ncols, title, fig_type):
    """
    Create custom subplots for KDE plots or box plots.

    Parameters:
    - df: DataFrame containing the data.
    - target: The target variable for the plot (if applicable).
    - ncols: Number of columns for subplots.
    - title: Title for the entire set of subplots.
    - fig_type: Type of plot to create, either 'kde' for KDE plots or 'boxplot' for box plots.

    Returns:
    - None
    """
    # Determine the columns to use for plotting based on the presence of a target variable
    if target:
        cols = df.drop(target, axis=1).columns
    else:
        cols = df.columns

    # Calculate the number of rows for subplots based on the number of columns and ncols
    nrows = len(cols) // ncols + (len(cols) % ncols > 0)

    # Create a figure
    plt.figure(figsize=(20, 30))
    plt.subplots_adjust(hspace=0.9)
    plt.suptitle(title, fontsize=15, y=0.95)

    for n, variable in enumerate(cols):
        # Add a new subplot iteratively
        ax = plt.subplot(nrows, ncols, n + 1)

        # Create KDE plots or box plots based on fig_type and the presence of a target variable
        if (fig_type == "kde") & (target is None):
            sns.kdeplot(ax=ax, x=variable, y=target, data=df, fill=True)
        elif fig_type == "boxplot":
            sns.boxplot(ax=ax, x=variable, y=target, data=df)
        else:
            raise ValueError("Enter a valid value for fig_type and target")

        # Customize subplot appearance
        ax.grid(b=True, which="major", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"Variable = {variable}", loc="center", fontsize=15)


def subplot_barplot(df, ncols, title):
    """
    Create subplots for barplot figures based on DataFrame columns.

    Parameters:
    - df: DataFrame containing the data for barplots.
    - ncols: Number of columns for subplots.
    - title: Title for the entire set of subplots.

    Returns:
    - None
    """
    # Calculate the number of rows for subplots based on the number of columns and ncols
    nrows = len(df.columns) // ncols + (len(df.columns) % ncols > 0)

    # Create a figure
    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(title, fontsize=9, y=0.95)

    for n, variable in enumerate(df.columns):
        # Add a new subplot iteratively
        ax = plt.subplot(nrows, ncols, n + 1)

        # Compute percentage of each value in the column
        temp = df[variable].value_counts(normalize=True).reset_index()
        temp.columns = [variable, "pct"]

        # Compute the volume (count) of each value in the column
        temp2 = df[variable].value_counts().reset_index()
        temp2.columns = [variable, "volume"]

        # Join percentage and volume data
        data = pd.concat([temp, temp2["volume"]], axis=1)

        # Sort the data by the variable in descending order
        data = data.sort_values(by=variable, ascending=False).reset_index()

        # Create a barplot using Seaborn
        ax = sns.barplot(
            x="pct", y=variable, data=data, palette="Paired", order=data[variable]
        )

        # Extract volume data for labeling
        vol = [list(data["volume"])]

        try:
            # Add volume labels next to the bars
            for container, number in zip(ax.containers, vol):
                ax.bar_label(container, labels=number)
        except:
            pass

        ax.grid(b=True, which="major", axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_title(variable, loc="center", fontsize=9)
        ax.set_ylabel(variable, fontsize=9)
        ax.set_xlabel("%", fontsize=9)


def plot_corr_matrix(df, title):
    """
    Create a correlation matrix heatmap plot using Plotly.

    Parameters:
    - df: DataFrame containing the data for the correlation matrix.
    - title: Title for the correlation matrix plot.

    Returns:
    - None
    """
    # Calculate the correlation matrix
    corrMatrix = df.corr()

    # Set the default Plotly template to "plotly_white"
    pio.templates.default = "plotly_white"

    # Create a heatmap using Plotly
    heat = go.Heatmap(
        z=corrMatrix,
        x=corrMatrix.columns,
        y=corrMatrix.columns,
        colorscale=px.colors.diverging.delta,
        zmin=-1,
        zmax=1,
    )

    # Define the layout for the heatmap plot
    layout = go.Layout(
        title_text=title,
        title_x=0.5,
        width=600,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )

    # Create a Figure and display the heatmap plot
    fig = go.Figure(data=[heat], layout=layout)
    fig.show()
    return corrMatrix
