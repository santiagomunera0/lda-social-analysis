
# Import libraries
import os
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from collections import Counter
import math
import pyLDAvis.gensim_models
from NLP import (
    local_reading,
    read_corpus,
    optimized_lda_classifier,
    lda_publications_classifier,
    json_write,
    load_gensim_lda_model,
)
from AI import topic_names_gpt
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors
import numpy as np

import json
import copy
from collections import defaultdict
import sklearn
import sklearn.feature_extraction
import networkx as nx


# Functions
def choerence_curve(model_results, k):
    """
    Plot and analyze the coherence curve for topic modeling.

    This function takes a DataFrame of model results, typically containing coherence scores for different
    numbers of topics (k), and plots the coherence curve. It highlights the recommended number of topics (k)
    based on the maximum coherence score.

    Args:
        model_results (DataFrame): A DataFrame containing model results with 'Coherence' scores.
        k (int): The recommended number of topics based on maximum coherence.

    Returns:
        None
    """
    x = model_results.index.tolist()
    y = model_results["Coherence"].tolist()
    ax = pd.DataFrame(model_results).plot(figsize=(14, 4))
    # pd.DataFrame({'coherence':y},index=x).sort_values(by='coherence',ascending=False).reset_index().rename({'index':'k'}).iloc[[0]].plot.scatter(x='index',y='coherence',ax=ax,color='r')
    point = (
        pd.DataFrame({"Coherence": y}, index=x)
        .sort_values(by="Coherence", ascending=False)
        .reset_index()
    )
    point[point["index"] == k].plot.scatter(x="index", y="Coherence", ax=ax, color="r")
    print(f"Recomended k: {k}")


def coherence_graphic_process(k: int, path_models="", name=""):
    """
    Generate and display a coherence graphic for a given number of topics (k) based on model results.

    This function reads model results from a local file, generates a coherence curve for different
    numbers of topics, and displays the curve. It helps determine the optimal number of topics for
    topic modeling.

    Args:
        k (int): The number of topics for which to generate the coherence curve.
        path_models (str): The directory path where the model results file is located.
        name (str): The name of the model results file.

    Returns:
        None
    """
    try:
        model_results = local_reading(path=os.path.join(path_models, name)).set_index(
            "Topics"
        )
        choerence_curve(model_results=model_results, k=k)
    except Exception as e:
        print("No File for coherence graphic")
        print(f"Error: {e}")


def topic_wordclouds(lda_model, path_figures: str, NUM_TOPICS: int, num_words=20):
    """
    Generate and display wordclouds for each topic in an LDA topic model.
    """

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    for i in range(NUM_TOPICS // 10):
        cols += [color for name, color in mcolors.TABLEAU_COLORS.items()]

    # Calculamos el número de filas y columnas necesarias
    num_rows = (
        NUM_TOPICS + 4
    ) // 5  # Esto calcula el número mínimo de filas necesarias
    num_cols = min(NUM_TOPICS, 5)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(30, 6 * num_rows), sharex=True, sharey=True
    )

    # Asegurarse de que 'axes' sea un arreglo bidimensional
    if num_rows == 1:
        axes = np.array([axes])
    if num_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i in range(NUM_TOPICS):
        cloud = WordCloud(
            stopwords=[],
            background_color="white",
            width=600,
            height=400,
            max_words=50,
            colormap="tab10",
            color_func=lambda *args, **kwargs: cols[i],
            prefer_horizontal=1.0,
        )

        topics = lda_model.show_topics(
            formatted=False, num_topics=NUM_TOPICS, num_words=num_words
        )
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words)

        row = i // num_cols
        col = i % num_cols

        ax = axes[row, col]
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title("Topic " + str(i), fontsize=16)
        ax.axis("off")

    # Eliminamos los ejes sobrantes si no se usan
    for j in range(NUM_TOPICS, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        fig.delaxes(axes[row, col])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(path_figures, f"topics{NUM_TOPICS}.png"))


def lda_words_by_relevance(
    df, path_figures: str, path_output: str, lda_model, NUM_TOPICS
):
    """
    Generate and display word importance and count plots for topics in an LDA topic model.

    This function analyzes and visualizes the importance and count of words for each topic in an LDA (Latent Dirichlet Allocation) topic model.

    Args:
        df (DataFrame): The input DataFrame containing processed text data.
        path_figures (str): The directory path for saving the figures.
        path_output (str): The directory path for saving the output data.
        lda_model: The LDA topic model object.
        NUM_TOPICS (int): The number of topics in the LDA model.

    Returns:
        ouput_df (Dataframe): data of lda_words_by_relevance.
        cols (list) : list of colors used for each topic.
    """

    df = df.dropna(subset=["processed"])
    data_ready = df["processed"].str.split(" ")
    topics = lda_model.show_topics(num_words=25, formatted=False, num_topics=NUM_TOPICS)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    output_df = pd.DataFrame(
        out, columns=["word", "topic_id", "importance", "word_count"]
    )
    output_df = output_df.sort_values(by=["topic_id", "importance"])
    twin_ylim_max = {}
    for i in range(NUM_TOPICS):
        max_val = output_df.loc[output_df.topic_id == i]["importance"].max()
        max_val /= 0.9
        max_val = math.ceil(max_val * 500) / 500
        twin_ylim_max[i] = max_val

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(
        (NUM_TOPICS + 1) // 2,
        2,
        figsize=(24, ((NUM_TOPICS + 1) // 2) * 8),
        sharex=True,
        dpi=160,
    )
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i in range(NUM_TOPICS // 10):
        cols += [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        if i >= NUM_TOPICS:
            break
        # Bars
        ax.barh(
            y="word",
            width="word_count",
            data=output_df.loc[output_df.topic_id == i, :],
            color=cols[i],
            height=0.8,
            alpha=0.5,
            label="Conteo de Palabras",
        )
        ax_twin = ax.twiny()
        ax_twin.barh(
            y="word",
            width="importance",
            data=output_df.loc[output_df.topic_id == i, :],
            color=cols[i],
            height=0.35,
            label="Pesos",
        )
        # Lims
        ax_twin.set_xlim(0, twin_ylim_max[i])
        ax.legend(loc="lower right")
        ax_twin.legend(loc="upper right")
        ax.xaxis.set_tick_params(labelbottom=True)

        # Labels
        # ax.set_title('Temática: ' + lista_temas[i], color=cols[i], fontsize=22)
        ax.set_xlabel("Conteo de Palabras", color=cols[i], fontsize=16)
        ax_twin.set_xlabel("Peso en la temática", color=cols[i], fontsize=16)
        ax.tick_params(axis="x", labelsize=14)
        ax_twin.tick_params(axis="x", labelsize=14)
        ax.set_yticklabels(
            output_df.loc[output_df.topic_id == i, "word"],
            fontsize=18,
            horizontalalignment="right",
        )

    output_df.to_csv(
        os.path.join(path_output, "words_global.csv"),
        index=False,
    )
    fig.tight_layout(w_pad=2)
    my_suptitle = fig.suptitle(
        "Conteo de Palabras e Importancia Relativa de las Palabras Clave por Temática",
        fontsize=22,
        y=1.05,
    )
    fig.savefig(
        os.path.join(path_figures, f"topics{NUM_TOPICS}_lda_words_by_relevance.png")
    )
    plt.show()

    return output_df, cols


def visualization_tsne(df, lda_model, path_models, NUM_TOPICS, output_df, cols):
    """
    Generate and display the visualization of the technique T-distributed stochastic
    neighbor embedding (T-SNE).

    This function uses the technique T-SNE to visualize the data output of the LDA
    in two dimensions to help analyze the relationship between topics.

    Args:
        df (DataFrame): The input DataFrame containing processed text data.
        lda_model: The LDA topic model object.
        path_models (str): The directory path where the LDA model is located.
        NUM_TOPICS (int): The number of topics to model.
        ouput_df (Dataframe): data of lda_words_by_relevance.
        cols (list) : list of colors used for each topic.

    Returns:
        plot_tsne: graphic to visualize
        output_json (json): data of LDA and T-SNE.
    """
    corpus = read_corpus(path_models)

    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    # arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=0.99, init="pca")
    tsne_lda = tsne_model.fit_transform(arr)
    tsne_lda = np.float64(tsne_lda)

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = NUM_TOPICS
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    for i in range(NUM_TOPICS // 10):
        mycolors = np.append(
            mycolors,
            np.array([color for name, color in mcolors.TABLEAU_COLORS.items()]),
        )
    plot_tsne = figure(
        title="t-SNE Clustering of {} LDA Topics".format(n_topics),
        width=900,
        height=700,
    )
    plot_tsne.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    # show(plot)

    lista_topics = []
    cont = 0
    for frase in corpus:
        cont += 1
        # print(cont)
        topic_percs, wordid_topics, wordid_phivalues = lda_model[frase]
        topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
        lista_topics.append(topic_percs_sorted[0][0])
    df["tematica"] = lista_topics

    output_tsne = []
    i = 0
    for _, row in df.iterrows():
        item = {}
        item["x"] = tsne_lda[i, 0]
        item["y"] = tsne_lda[i, 1]
        item["color"] = mycolors[row["tematica"]]
        item["topic"] = row["tematica"]
        item["text"] = row["content"]
        item["prop"] = row["processed"]
        item["date"] = row["date"]

        output_tsne.append(copy.deepcopy(item))
        i += 1
    # output_df = pd.read_csv(path_output+'words_global.csv')
    output_weights = json.loads(output_df.to_json(orient="records"))
    output_json = {
        "LDA": {
            "TSNE": output_tsne,
            "topic_weights": output_weights,
            "colors_topics": cols,
        }
    }
    return plot_tsne, output_json


def co_occurrence_directed(sentences):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split(" ")
        # iterate over sentences
        for i in range(len(text) - 1):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i + 1]
            vocab.add(next_token)
            key = tuple([token, next_token])
            # if(token == 'vida' and next_token == 'cotidiano'):
            # print(text)
            # print(key)
            d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab)  # sort vocab
    df = pd.DataFrame(
        data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
        index=vocab,
        columns=vocab,
    )
    return d


def dfs_recursive(g, node, nodes_color, edges_to_be_removed):
    global found
    global node_found
    # print('entering',node,nodes_color[node])
    nodes_color[node] = 1
    childs = np.random.permutation([*g.successors(node)])
    # print(nodes_color)
    for child in childs:
        # print(node,child)
        if found:
            # print('found',child)
            break
        elif nodes_color[child] == 0:
            # print('exploring',child)
            dfs_recursive(g, child, nodes_color, edges_to_be_removed)
            if found and node_found is not None:
                edges_to_be_removed.append((node, child, g[node][child]["weight"]))
                if node_found == node:
                    node_found = None
        elif nodes_color[child] == 1:
            # print('exploring cycle',child)
            found = True
            node_found = child
            edges_to_be_removed.append((node, child, g[node][child]["weight"]))

    # print('exiting',node,nodes_color[node])
    nodes_color[node] = 2


def dfs_remove_edges(g):
    global found
    global node_found
    found = False
    node_found = None
    nodes_color = {}
    edges_to_be_removed = []
    for node in g.nodes():
        nodes_color[node] = 0
    nodes = np.random.permutation(g.nodes())

    for node in nodes:

        if found:
            break
        if nodes_color[node] == 0:
            dfs_recursive(g, node, nodes_color, edges_to_be_removed)
    return edges_to_be_removed


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def graph_process(df, output_json, NUM_TOPICS, path_output=""):
    """
    It generates an acyclic directed graph for each topic where the nodes
    are the tokens of each topic, and then stores the statistical data of
    the relationship between nodes (tokens), the frequency of each token in
    the topic and the topological order of the graph.

    Args:
        df (DataFrame): The input DataFrame containing processed text data.
        NUM_TOPICS (int): The number of topics to model.
        output_json (json): data of LDA and T-SNE.
        path_output (str): The directory path for saving the output data.

    Returns:
        None
    """

    df["processed"] = df["processed"].fillna("<EMPTY>")
    export_tematica = []
    for i in range(NUM_TOPICS):
        df_tematica = df[df.tematica == i]
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            max_features=500, binary=True
        )
        tdm_sk = vectorizer.fit_transform(df_tematica["processed"]).T

        similarity = sklearn.metrics.pairwise.cosine_similarity(tdm_sk)
        edges = []

        G = nx.Graph()
        G_inv = nx.Graph()
        for i in range(len(similarity)):
            for j in range(i + 1, len(similarity)):
                if similarity[i, j] > 0:
                    edges.append({"source": i, "target": j, "weight": similarity[i, j]})
                    G.add_edge(i, j, weight=similarity[i, j])
                    G_inv.add_edge(i, j, weight=1 / similarity[i, j])
        nodes = [
            {
                "word": word_id[0],
                "id": word_id[1],
                "freq": freq / df_tematica.shape[0],
                "closeness": closeness[1],
                "betweenness": betweenness[1],
            }
            for word_id, freq, closeness, betweenness in zip(
                sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]),
                np.count_nonzero(tdm_sk.toarray(), axis=1),
                sorted(nx.closeness_centrality(G_inv).items()),
                sorted(nx.betweenness_centrality(G_inv).items()),
            )
        ]
        for idx, val in enumerate(
            np.argsort(np.array([node["freq"] for node in nodes]))
        ):
            nodes[idx]["rank"] = val
        export_tematica.append(
            {"nodes": copy.deepcopy(nodes), "edges": copy.deepcopy(edges)}
        )
    output_json["CORR"] = export_tematica

    top_story = []
    for i in range(NUM_TOPICS):
        df_tematica = df[df.tematica == i]
        D = co_occurrence_directed(df_tematica.processed)
        G_directed = nx.DiGraph()
        for row in D.items():
            G_directed.add_edge(row[0][0], row[0][1], weight=row[1])
        G_directed.remove_edges_from(nx.selfloop_edges(G_directed))
        edges = sorted(
            G_directed.edges(data=True), key=lambda x: x[2]["weight"], reverse=True
        )
        G_top = nx.DiGraph()
        G_top.add_edges_from(edges[: math.ceil(math.sqrt(len(edges)))])
        while not nx.is_directed_acyclic_graph(G_top):

            e = dfs_remove_edges(G_top)
            edge_to_remove = sorted(e, key=lambda x: x[2])[0]
            G_top.remove_edge(edge_to_remove[0], edge_to_remove[1])

        nx.is_directed_acyclic_graph(G_top)
        top_story.append({"topic": i, "story": [*nx.topological_sort(G_top)]})
    output_json["top_story"] = top_story
    with open(path_output + "visualization.json", "w") as outfile:
        json.dump(output_json, outfile, cls=NpEncoder)


def dict_tildes(txt):
    """
    Transform a word removing the accents.

    This function take a token (word) and replace the letters with tilde
    to the same letters without tilde

    Args:
        txt (str): text, commonly a token which is a word.

    Returns:
        txt (str)
    """
    diccionario_tildes = {
        # "á" : "%C3%A1",
        # "é" : "%C3%A9",
        # "í" : "%C3%AD",
        # "ó" : "%C3%B3",
        # "ú" : "%C3%BA",
        # "ñ" : "%C3%B1"}
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "ñ",
    }

    transTable = txt.maketrans(diccionario_tildes)
    txt = txt.translate(transTable)
    return txt


def data_words_topics(path_output=""):
    """
    Create and save the dataframe of the words of each topic.

    First crate and save a dataframe that has 'date', 'topic' and count the rows of that
    topic in that date and save it in a column.In addition, it creates a dataframe with
    each word that has each text and its columns are the date, the subject, the
    processed text and the word. It also creates a dataframe for each topic
    with the texts containing at least one of the 6 most frequent words, with the date
    and the processed text.

    Args:
        path_output (str): The directory path for saving the output data.

    Returns:
        None
    """
    # Se añade el path_output tanto pa leer el classified.csv como pa guardar el timeline_topics...
    data = pd.read_csv(
        os.path.join(
            path_output,
            "df_classified.csv",
        )
    )
    data[["date", "Topic", "author"]].groupby(
        ["date", "Topic"]
    ).count().reset_index().to_csv(
        os.path.join(
            path_output,
            "timeline_topicsAll.csv",
        ),
        index=False,
    )

    df_palabras_tematica = data[["date", "Topic", "processed"]].copy()
    df_palabras_tematica["Tags"] = (
        df_palabras_tematica["processed"].str.split(" ").tolist()
    )
    df_palabras_tematica = df_palabras_tematica.dropna()

    df_fin = []
    for index, row in df_palabras_tematica.iterrows():
        if len(row["Tags"]) > 0:
            lista = row["Tags"]
            for l in lista:
                copy_row = row.copy()
                copy_row["Palabra"] = l
                df_fin.append(copy_row)

            else:
                copy_row = row.copy()
                copy_row["Palabra"] = ""
                df_fin.append(copy_row)
    df_fin = pd.DataFrame(df_fin)
    del df_fin["Tags"]
    df_fin.to_csv(
        os.path.join(
            path_output,
            "Words_Topic_Timeline.csv",
        ),
        index=False,
    )
    for _ in list(df_fin["Topic"].unique()):
        df_aux = df_fin[df_fin["Topic"] == _].copy()
        df_aux["Palabra"] = df_aux["Palabra"].apply(lambda x: dict_tildes(x))
        df_aux = df_aux[df_aux["Palabra"] != ""]
        df_most_sign_word = (
            df_aux[["Palabra", "Topic"]]
            .groupby("Palabra")
            .count()
            .reset_index(drop=False)
            .sort_values(by=["Topic"], ascending=False)
            .head(6)
        )
        palabras = list(df_most_sign_word["Palabra"].unique())
        df_aux = df_aux[df_aux["Palabra"].isin(palabras)]
        df_aux.to_csv(
            os.path.join(
                path_output,
                f"words_timeline_topic_{_}.csv",
            ),
            encoding="utf-8",
            index=False,
        )


def Clasification_Vizualization_lda_process(
    df,
    NUM_TOPICS: int,
    path_models,
    path_figures,
    num_words=20,
    optimization=True,
    Topic_assistant=False,
    assistant_key="",
    path_output="",
    contexto="",
):
    """
    Perform topic modeling, classification, and visualization of text data using LDA.

    This function performs the following tasks:
    1. Loads a pre-trained LDA (Latent Dirichlet Allocation) topic model.
    2. Performs topic classification on the text data.
    3. Generates and displays word clouds for each topic.
    4. Analyzes and visualizes the importance and count of words for each topic.
    5. Creates interactive visualization using PyLDAvis.
    6. Visualize the T-SNE and save its data in json.
    7. Create a graph for each topic and save its data in json.
    8. Saves the results and distributions to CSV files.
    9. Save the data of the words most frequented of each topic to CSV files.
    10. Optionally, generates topic names using OpenAI GPT-3 for topic labeling.

    Args:
        df (DataFrame): The input DataFrame containing text data.
        NUM_TOPICS (int): The number of topics to model.
        path_models (str): The directory path where the LDA model is located.
        path_figures (str): The directory path for saving figures.
        num_words (int): The number of top words to display in word clouds (default is 20).
        optimization (bool): Perform optimization when classifying topics (default is True).
        Topic_assistant (bool): Use OpenAI GPT-3 for topic labeling (default is False).
        assistant_key (str): API key for OpenAI GPT-3 (required if Topic_assistant is True).
        path_output (str): The directory path for saving output data (optional).
        contexto (str): Additional context for topic labeling with GPT-3 (optional).

    Returns:
        df (DataFrame): The input DataFrame with topic classification results.
        lda_model: The loaded LDA topic model.
    """

    lda_model = load_gensim_lda_model(path_models=path_models, NUM_TOPICS=NUM_TOPICS)

    corpus = read_corpus(path_models)
    print(
        "Clasification process"
    )  # WHE CLASIFICATION IS DONE AFTER TSNE , THE PROCESESS NEVER END.
    clasification_proba = optimized_lda_classifier(
        corpus=corpus, lda_model=lda_model, n_cpus=None, optimization=optimization
    )
    if optimization:
        clasification_proba = pd.concat(clasification_proba, axis=0).reset_index(
            drop=True
        )
    df = pd.concat([df, clasification_proba], axis=1)
    df.to_csv(
        os.path.join(
            path_output,
            "df_classified.csv",
        )
    )
    df["Topic"].value_counts().to_csv(
        os.path.join(
            path_output,
            "topic_distribution.csv",
        )
    )

    topic_wordclouds(
        lda_model=lda_model,
        path_figures=path_figures,
        NUM_TOPICS=NUM_TOPICS,
        num_words=num_words,
    )

    output_df, cols = lda_words_by_relevance(
        df=df,
        path_figures=path_figures,
        path_output=path_output,
        lda_model=lda_model,
        NUM_TOPICS=NUM_TOPICS,
    )

    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(
        lda_model,
        corpus,
        dictionary=lda_model.id2word,
        sort_topics=False,
    )
    pyLDAvis.save_html(
        vis, os.path.join(path_figures, f"topics{NUM_TOPICS}_pyldavis.html")
    )
    display(vis)

    """figure_tsne, output_json = visualization_tsne(df=df,lda_model=lda_model, path_models=path_models, NUM_TOPICS=NUM_TOPICS, output_df=output_df, cols=cols)
    show(figure_tsne)
    output_file(filename="tsne.html", title="tsne")
    save(figure_tsne)"""

    # graph_process(df=df,output_json=output_json, NUM_TOPICS=NUM_TOPICS, path_output=path_output)#process of . the output was required by a proces of Rafael mantilla (web page implementation- is in standby). Hevy process

    data_words_topics(path_output)  # Heavy process
    if Topic_assistant and assistant_key:
        print("Open AI API")
        topics_names = topic_names_gpt(
            lda_model=lda_model,
            NUM_TOPICS=NUM_TOPICS,
            key=assistant_key,
            contexto=contexto,
        )
        print(topics_names)
        json_write(os.path.join(path_output, "topic_names.json"), topics_names)
    else:
        print("Not provided Topic_assistant or assistan_key")
    return df, lda_model
