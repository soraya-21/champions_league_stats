import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
from sklearn.linear_model import LinearRegression

# --- Charger les données ---
df = pd.read_csv("all_players_clean.csv")

def vis_PHY_DEF_by_nation(df):
    st.title("Visualisation de PHY et DEF par Nation")
    # --- Garder les 10 nations les plus fréquentes ---
    top_nations = df['Nation'].value_counts().nlargest(10).index
    df_top = df[df['Nation'].isin(top_nations)]

    # --- Option de visualisation ---
    option = st.selectbox("Que voulez-vous visualiser ?", ["Une seule stat", "PHY et DEF côte à côte"])

    if option == "Une seule stat":
        stat = st.selectbox("Choisir la stat à visualiser", ["PHY", "DEF"])
        color = "purple" if stat == "PHY" else "yellow"

        fig = sns.catplot(
            data=df_top,
            x="Nation",
            y=stat,
            kind="box",
            height=5,
            aspect=2,
            color=color
        )
        fig.fig.suptitle(f"Distribution de {stat} par Nation", y=1.05)
        plt.xticks(rotation=45)
        st.pyplot(fig.fig)

    else:  # PHY et DEF côte à côte
        # Melt pour transformer PHY et DEF en une colonne "Stat" et une colonne "Valeur"
        df_melt = df_top.melt(id_vars=["Nation"], value_vars=["PHY", "DEF"],
                            var_name="Stat", value_name="Valeur")
        
        # Palette pour distinguer PHY et DEF
        palette = {"PHY": "purple", "DEF": "yellow"}

        fig = sns.catplot(
            data=df_melt,
            x="Nation",
            y="Valeur",
            hue="Stat",
            kind="box",
            height=5,
            aspect=2,
            palette=palette
        )
        fig.fig.suptitle("Distribution de PHY et DEF par Nation", y=1.05)
        plt.xticks(rotation=45)
        st.pyplot(fig.fig)


def compare_PAC_DRI_SHO_between_leagues(df):
    st.title("Comparaison de PAC, DRI et SHO entre les Championnats")
    championnats = st.multiselect(
    "Choisir 3 championnats",
    options=df["League"].unique(),
    default=df["League"].unique()[:3]
    )

    df_top = df[df["League"].isin(championnats)]

    stats = ["PAC", "DRI", "SHO"]

    df_melt = df_top.melt(id_vars=["League"], value_vars=stats, var_name="Stat", value_name="Valeur")

    fig = px.box(
        df_melt,
        x="League",
        y="Valeur",
        color="Stat",
        boxmode="group",
        title="Comparaison des statistiques PAC, DRI et SHO par championnat"
    )

    st.plotly_chart(fig, use_container_width=True)

def compare_PAC_SHO_with_regression(df):
    st.title("Comparaison de PAC et SHO avec régression par League")

    all_leagues = sorted(df["League"].dropna().unique())

    selected_leagues = st.multiselect(
        "Choisis les leagues à afficher :",
        all_leagues,
        default=all_leagues[:3]
    )

    if not selected_leagues:
        st.warning("Sélectionne au moins une league.")
        st.stop()

    filtered_df = df[df["League"].isin(selected_leagues)]

    p = figure(
        title="PAC vs SHO — Sélection par league (avec trendline)",
        x_axis_label="PAC",
        y_axis_label="SHO",
        width=800,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,hover"
    )

    palette = Category10[10]
    color_map = {league: palette[i % len(palette)] 
                for i, league in enumerate(selected_leagues)}

    for league in selected_leagues:
        league_data = filtered_df[filtered_df["League"] == league]

        # Scatter points
        source = ColumnDataSource(league_data)
        p.circle(
            x="PAC",
            y="SHO",
            source=source,
            size=7,
            alpha=0.6,
            color=color_map[league],
            legend_label=f"{league} (points)",
        )

        # ---------------------------
        # TRENDLINE (linear regression)
        # ---------------------------
        x = league_data["PAC"].values
        y = league_data["SHO"].values

        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)

            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = slope * x_line + intercept

            p.line(
                x_line,
                y_line,
                line_width=2,
                color=color_map[league],
                alpha=0.9,
                legend_label=f"{league} (trendline)",
            )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    st.bokeh_chart(p, use_container_width=True)

def compare_players_by_stat(df):
    st.title("Comparaison radar entre deux joueurs par poste")

    # 2. Choisir le poste
    positions = sorted(df["Position"].dropna().unique())
    selected_position = st.selectbox("Choisis le poste :", positions)

    df_pos = df[df["Position"].str.contains(selected_position, na=False)]

    if df_pos.empty:
        st.error(f"Aucun joueur trouvé pour le poste {selected_position}")
        st.stop()

    # 3. Top 2 joueurs du poste
    top_players_df = df_pos.sort_values("OVR", ascending=False).head(2)
    top_players = top_players_df["Name"].tolist()

    st.success(f"Top 2 {selected_position} : {top_players[0]} et {top_players[1]}")

    # 4. Ajouter d'autres joueurs
    all_names = sorted(df_pos["Name"].dropna().unique())
    additional_players = st.multiselect(
        "Ajouter d'autres joueurs à comparer :",
        options=[name for name in all_names if name not in top_players]
    )

    players_to_compare = top_players + additional_players

    # 5. Préparer les données pour radar
    stats = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]
    labels = stats
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # 6. Création du radar plot
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = plt.cm.get_cmap("tab10", len(players_to_compare))

    for i, name in enumerate(players_to_compare):
        player = df_pos[df_pos["Name"] == name].iloc[0]
        values = player[stats].values
        values = np.concatenate((values, [values[0]]))  # fermer le radar
        
        ax.plot(angles, values, linewidth=2, label=name, color=colors(i))
        ax.fill(angles, values, alpha=0.15, color=colors(i))

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_title(f"Comparaison radar — Poste : {selected_position}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # 7. Affichage dans Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    vis_PHY_DEF_by_nation(df)
    compare_PAC_DRI_SHO_between_leagues(df)
    compare_PAC_SHO_with_regression(df)
    compare_players_by_stat(df)