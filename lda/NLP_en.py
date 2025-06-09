# Import
from multiprocessing import Pool, cpu_count
import polars as pl
import pandas as pd
import os
from tqdm import tqdm
import re
import emoji
import numpy as np
import json
import pickle
import time
from datetime import datetime

# npl
from nltk.corpus import stopwords
from nltk.data import find
from nltk import download
from gensim.utils import simple_preprocess
import spacy
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaMulticore
from kneed import KneeLocator
import pyLDAvis.gensim_models


# functions
def create_folder(path_folder):
    """
    Create a folder at the specified path if it does not already exist.

    Args:
        path_folder (str): The path of the folder to be created.

    Returns:
        None

    Raises:
        OSError: If an error occurs while attempting to create the folder.
    """
    try:
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
            print(f"The folder {path_folder} has been created.")
        else:
            print(f"The folder {path_folder} already exists.")
    except OSError as e:
        print(f"Error creating folder {path_folder}: {str(e)}")


def local_reading(path, format="csv", sep=",", lib="pandas"):
    """Read files.

    Read one or several  comma-separated values (csv)
    file into DataFrame with polars or pandas dataframe.

    Args:
      path: str, required
        Sting with the path to the folder or the file.
      format: str, default csv
        If the path is a folder only reads files with the desire format.
      sep:str, default ','
         Delimiter to use.
      lib: {'pandas', 'polars'} default pandas
        only por pandas or polars
    require_all_keys:
        If True only rows with values set for all keys will be returned.

    Returns:
      A Dataframe object (pandas or polars)

    Raises:
      IOError: An error occurred accessing the path.
    """
    if lib == "polars":
        if os.path.isfile(path):
            file = pl.read_csv(path, separator=sep, infer_schema_length=0)
            return file
        if os.path.isdir(path):
            files = [i for i in os.listdir(path) if i.endswith(format)]  # list o files
            # dfs = list(map(lambda x: pl.read_csv(path+'/'+x,separator=sep,infer_schema_length=0), tqdm(files,leave=True)))#list of frames

            # transform empty str to null
            dfs = list(
                map(
                    lambda x: pl.read_csv(
                        path + "/" + x, separator=sep, infer_schema_length=0
                    ).with_columns(
                        [
                            pl.when(pl.col(pl.Utf8).str.lengths() == 0)
                            .then(None)
                            .otherwise(pl.col(pl.Utf8))
                            .keep_name()
                        ]
                    ),
                    tqdm(files, leave=True, desc="Data Reading"),
                )
            )
            return pl.concat(dfs)
    elif lib == "pandas":

        if os.path.isfile(path):
            file = pd.read_csv(path, sep=sep)
            return file

        if os.path.isdir(path):
            files = [i for i in os.listdir(path) if i.endswith(format)]  # list o files
            dfs = list(
                map(
                    lambda x: pd.read_csv(path + "/" + x, sep=sep),
                    tqdm(files, leave=True),
                )
            )  # list of frames
            return pd.concat(dfs)
    else:
        print("Is not pandas or polars library")


def check_stopwords_resource():
    """Check if the stopword resource is already downloaded, if not, it downloads."""
    if not download("stopwords", quiet=True):
        print("Stopwords are already downloaded.")
    else:
        print("Stopwords have been downloaded.")


def normalize_columns(df, dir_names):
    """Change names and filter data columns.

    Change the names of the data columns and filter only by the selected ones:
    with a directory, select the necessary columns and the new name of the column.
    Only the columns in the Directory are used.

    Args:
        df: pandas or polars dataframe.
            Dataframe
        dir_names: dict
            Directory that contains the names of the dataframe columns as keys and the new names as values.

    Returns:
      A Dataframe object (pandas or polars)

    Raises:
      IOError: An error occurred accessing the path.
    """
    if isinstance(df, pd.DataFrame):
        df = df.loc[:, dir_names.keys()]  # Filter by columns needed
        df = df.rename(columns=dir_names)  # Rename columns
        print("Is pandas Dataframe")
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.select(dir_names.keys())  # Filter by columns needed
        df = df.rename(dir_names)  # Rename columns
        print("Is polars Dataframe")
        return df
    else:
        print("Is not pandas or polars library")


def Network_ditribution(df, path_output=""):
    """Network distribution.

    Write a csv file with the network distribution. Only for pandas and polars dataframe.

    Args:
        df: pandas or polars dataframe.
        path_output: str, default ''
            String with the path to the output folder.

    Returns:
      No return.

    Raises:

    """
    if isinstance(df, pd.DataFrame):

        df["domain_group"].value_counts(ascending=False).to_csv(
            path_output + "network_distribution_pandas.csv", index=True
        )
    elif isinstance(df, pl.DataFrame):
        df["domain_group"].value_counts(sort=True).write_csv(
            path_output + "network_distribution_polars.csv", has_header=True
        )

    else:
        print("Is not pandas or polars library")


def Repeated_posts(df, path_output=""):
    """Calculate duplicated publications for pandas or polars dataframe.

    Write a csv file with the most repeated posts. Only for pandas and polars dataframe.

    Args:
        df: pandas or polars dataframe.
        path_output: str, default ''
            String with the path to the output folder.

    Returns:
    No return.

    Raises:

    """
    if isinstance(df, pd.DataFrame):
        df["content"].value_counts(ascending=False).to_csv(
            path_output + "Repeated_posts_pandas.csv", index=True
        )
    elif isinstance(df, pl.DataFrame):
        df["content"].value_counts(sort=True).write_csv(
            path_output + "Repeated_posts_polars.csv", has_header=True
        )

    else:
        print("Is not pandas or polars library")


def extract_hash(text):
    """Get Hashtagas in text.

    Create a list of Hashtags from a string. Delete  new line characters and extract the hashtags.

    Args:
        text: str
        text element like a publication.


    Returns:
        list of hashtags.

    Raises:

    """
    text = text.replace("\n", " ")
    return re.findall(r"#\w+", text)


def hash_cal(hash):
    """Transforms a list of hashtag lists into a pandas DataFrame.

    Clean every element in the list of list and count the ocurrence of the same elements.

     Args:
        hash : List
            list of hashtag lists

    Returns:
        A Pandas DataFrame containing the Hashtags and volume of each one.

    Raises:

    """

    lh = []
    for h in hash:
        lh.extend(h)
    chars_to_remove = [".", ",", "?", ""]
    regular_expression = "[" + re.escape("".join(chars_to_remove)) + "]"
    df_hashs = pd.DataFrame(lh, columns=["hashs"])
    df_hashs["hashs"] = df_hashs["hashs"].str.replace(
        regular_expression, "", regex=True
    )
    df_hashs["Conteo"] = 1
    Conteo_hashs = (
        df_hashs.groupby("hashs").sum().sort_values(by="Conteo", ascending=False)
    )
    return Conteo_hashs


def hash_table(df, path_output):
    """From a pandas or polars dataframe write a csv file containing the Hashtags and ist volumen.

    Filter files only containing # and process every publication with extract_hash,
    create a dataframe with hash_cal and wite the final dataframe as csv..


    Args:
        df: pandas or polars dataframe.
        path_output: str, default ''
        String with the path to the output folder.

    Returns:
    No return.

    Raises:

    """

    if isinstance(df, pd.DataFrame):

        hashs = (
            df[df["content"].apply(lambda x: "#" in x)]["content"]
            .apply(lambda x: extract_hash(x))
            .tolist()
        )
        hash_cal(hashs).to_csv(path_output + "Hash_global_pandas.csv", index=True)

    elif isinstance(df, pl.DataFrame):

        hashs = (
            df.filter(pl.col("content").str.contains(r"#"))["content"]
            .apply(lambda x: extract_hash(x))
            .to_list()
        )
        hash_cal(hashs).to_csv(path_output + "Hash_global_polars.csv", index=True)
    else:
        print("Is not pandas or polars library")


def drop_nan(df):
    """Drop  nan for polar and pandas.

    Drop  drop rows
    containing nan in the content column.

    Args:
        df: pandas or polars dataframe.

    Returns:
        pandas or polars dataframe.

    Raises:

    """

    if isinstance(df, pd.DataFrame):
        df.dropna(subset=["content"], inplace=True)

        print("Is pandas Dataframe")
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.filter(pl.col("content").is_not_null())

        print("Is polars Dataframe")
        return df
    else:
        print("Is not pandas or polars library")


def drop_duplicated_nan(df, column="content"):
    """Drop duplicated and nan for polar and pandas.

    Drop rows with duplicate items in the content column and drop rows
    containing nan in the content column.

    Args:
        df: pandas or polars dataframe.

    Returns:
        pandas or polars dataframe.

    Raises:

    """

    if isinstance(df, pd.DataFrame):
        df.dropna(subset=[column], inplace=True)
        df.drop_duplicates(subset=[column], inplace=True)
        print("Is pandas Dataframe")
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.filter(pl.col(column).is_not_null())
        df = df.unique(subset=[column])

        print("Is polars Dataframe")
        return df
    else:
        print("Is not pandas or polars library")


def columns_dtypes(df):
    """Set data type for columns. For the momento only for the columns containing time.

    Change data type  to pandas datatime  for the column containing date information.

    Args:
        df: pandas or polars dataframe.

    Returns:
        pandas or polars dataframe.

    Raises:

    """
    if isinstance(df, pd.DataFrame):
        df["date"] = pd.to_datetime(df["date"])
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.col("date").str.to_datetime(format="%Y-%m-%d %H:%M:%S"))
        return df
    else:
        print("Is not pandas or polars library")


def filter_columns_by_content(df, filter_list, name=""):
    """Filter DataFrame (Polars or pandas) by elements of a columns.

    Filter DataFrame by a list of elements in a specific column.

    Args:
        df: pandas or polars dataframe.
        filter_list: List,
            Social network to filter.

    Returns:
        Filtered pandas or polars dataframe.

    Raises:

    """
    if isinstance(df, pd.DataFrame):
        df = df[df[name].isin(filter_list)]
        print(df[name].unique())
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.filter(pl.col(name).is_in(filter_list))
        print(df.select([name]).unique())
        return df


# NPL


def sentences_to_words(sentences):
    """Sentences processing.

    String cleanup in multiple steps until the remaining items are relevant:
        *All words are transformed to lowercase.
        *Emails are removed.(There are publications with same text but only the email is different).
        *New line char are removed.
        *Remove single quotes.
        *Remove URLs.(There are publication with same text and different url).
        *Remove stock market tickers like $GE.
        *Remove old style retweet text "RT".
        *Remove words starting with # or @ (or different element if needed).
        *Remove emojis, pictograms, transport & map symbols,flags (iOS).
        *Translate emojis to words. (Emojis are also a form of expression in social networks).
        *Remove multiple white spaces.
        *Sequentially repeated elements are eliminated. Only one element remain.
        *Remove multiple white spaces .
        *Tokenize text into words and remove accent.

    Note: not all the steps are require for some project or maybe more steps are required.
    Add or silence depending on the project.

    Args:
        sentences: String element.

    Returns:
        clean string element.

    Raises:

    """

    sentences = re.sub("\\S*@\\S*\\s?", "", sentences)  # remove emails
    sentences = re.sub("\\s+", " ", sentences)  # remove newline chars
    sentences = re.sub("'", "", sentences)  # remove single quotes
    sentences = re.sub(
        r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", sentences
    )  # remove urls
    sentences = re.sub(r"\$\w*", "", sentences)  # remove stock market tickers like $GE
    sentences = re.sub(r"^RT[\s]+", "", sentences)  # remove old style retweet text "RT"
    sentences = sentences.lower()  # lower case
    # sentence  = re.sub("[@#][^\t\n\r\f\v\s]*"," ", sentence)#remove words starting with # or @ jose+

    # Remove accents mainly for spanish. simple_prprocess also remove accent.
    sentences = re.sub("á", "a", sentences)
    sentences = re.sub("é", "e", sentences)
    sentences = re.sub("í", "i", sentences)
    sentences = re.sub("ó", "o", sentences)
    sentences = re.sub("ú", "u", sentences)
    sentences = re.sub("ü", "u", sentences)

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )

    # sentences = emoji_pattern.sub(r'', sentences) # no emoji remove emoji
    sentences = emoji.demojize(sentences, language="en")
    # sentences = emoji.replace_emoji(sentences,' ')#delete emojis
    regex = "[\\¡\\!\\\"\\$\\%\\&\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\[\\\\\\]\\`\\{\\|\\}\\~]"  # Especial characters.
    sentences = re.sub(regex, " ", sentences)
    sentences = re.sub(r"\\s+", " ", sentences)  # remove multiple white spaces
    sentences = re.sub(
        r"\b(\w+)( \1\b)+", r"\1", sentences
    )  # Sequentially repeated elements
    sentences = simple_preprocess(
        str(sentences), deacc=True
    )  # tokenizes text. It converts a document into a list of lowercase tokens, while ignoring tokens that are too short or too long1. The deacc=True parameter removes accent marks from tokens using the deaccent() function1. For example, ‘é’ becomes just ‘e’.
    return sentences


def process_words(
    texts,
    stop_words=[],
    bigram_mod=None,
    allowed_postags=["NOUN", "ADJ", "VERB", "ADV", "PROPN", "NUM"],
):
    """Text processing
    From a pandas column containing text Remove Stopwords, Form Bigrams, Trigrams and Lemmatization.

    Args:
        text: Iterable element. pandas column, list
            Iterable element containing text in every element like a pandas column where every
            row is text or a list of texts.
        stop_words: list, default []
            List containing words for removal.
        bigram_mod: gensim.models.phrases, default None
            bigram model of the corpus. If None, it does not implement the birgam_mod.
        allowed_postag:list  default, ['NOUN', 'ADJ', 'VERB', 'ADV','PROPN','NUM']
            part-of-speech(POS) tag that are allowed to lemmatize.
    Return:
        list of text processed

    Raises:
    """
    if bigram_mod:
        texts = [
            [word for word in simple_preprocess(str(doc)) if word not in stop_words]
            for doc in texts
        ]  # Stopwords removal
        texts = [bigram_mod[doc] for doc in texts]  # Bigrams
        # texts = [trigram_mod[bigram_mod[doc]] for doc in texts]#Trigrams

    texts_out = []
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )  # Lemmatization
    # Remove stopwords once more after lemmatization
    texts_out = [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts_out
    ]  # 2nd Stopwords removal

    return texts_out


def denom(df, column_name="content"):
    """Multithreaded helper function to implement cleanup.

    Helper function that implement sentence_to_words function in a dataframe column.

    Args:
        df: Pandas dataframe.
        column_name: str, default content
            Name of the column in a dataframe containing text.

    Return:
        df: pandas dataframe with a column named processed containing the processed column.

    Raise:
    """
    df["processed"] = df[column_name].map(sentences_to_words)
    return df


def denom_process_words(df, stop_words=[], column_name="processed", bigram_mod=None):
    """Multithreaded helper function to implement tex processing with parameter.

    Helper function that implement process_words function in a dataframe column.
    This helper function pass parameters to the function.

    Args:
        df: Pandas dataframe.
        stop_words: list, default []
            List containing words for removal.
        column_name: str, default content
            Name of the column in a dataframe containing text.
        bigram_mod: gensim.models.phrases, default None
            bigram model of the corpus. If None, it does not implement the birgam_mod.

    Return:
        df: pandas datafraframe with a column named processed containing the processed column.

    Raise:
    """
    df[column_name] = process_words(df[column_name], stop_words, bigram_mod)
    return df


def parallelize_dataframe(df, func, n_cores=cpu_count(), **kwargs):
    """Function for multiprocess implementation.

    Uses the library multiprocessing for better use of the machine.
    Split the pandas dataframe for  multiple processors.
    For functions with only the split element is used pol.map.
    For functions that require more parameters o changing parameters is used pool.apply_async.

    Args:
        df: Pandas or polars dataframe.
        func : function,
            Function that is used in multiple process.
        n_cores: int, default cpu_count() maximum number of cpus.
            Number of cores to use. by default is used the maximum of the machine.
        kwargs:['stop_words','bigram_mod'],
            If kwargs are passed used apply_async. Only is passed stop_words and bigram_mod , for text processing.
    Return:
        processed df polars or pandas.

    Raise:
    """
    if isinstance(df, pd.DataFrame):

        print("Is pandas Dataframe")
        type_df = "Pandas"

    elif isinstance(df, pl.DataFrame):
        # Note: for the moment for multiprocess for polars is needed tu transform to pandas. Its important to find a way to multiprocess UDF in polars.
        df = df.to_pandas()
        print("Is polars Dataframe")
        type_df = "Polars"

    else:
        print("Is not pandas or polars library")

    pool = Pool(n_cores)

    df_split = iter(np.array_split(df, n_cores))
    del df
    if kwargs:
        async_results = [
            pool.apply_async(
                denom_process_words,
                args=(i, kwargs["stop_words"], "processed", kwargs["bigram_mod"]),
            )
            for i in df_split
        ]
        df = pd.concat([ar.get() for ar in async_results])

    else:
        df = pd.concat(
            pool.map(
                func,
                df_split,
            )
        )

    pool.close()
    pool.join()

    if type_df == "Polars":
        df = pl.from_pandas(df)
    return df


# LDA


def compute_coherence_values(
    corpus,
    dictionary,
    k,
    text,
    id2word,
    alpha=1,
    eta="auto",
    path_output="",
    lda_cpus=None,
):
    """Calculate Coherence and save  LDA model.

    Create a LDA model for the k selected, calculate de coherence model and save the model.
    Most of the parameters are for LdaMulticore and CoherenceModel from gensim library.
    ldaMulticore   : https://radimrehurek.com/gensim/models/ldamulticore.html.
    coherencemodel : https://radimrehurek.com/gensim/models/coherencemodel.html
    alpha and beta : https://www.thoughtvector.io/blog/lda-alpha-and-beta-parameters-the-intuition/

    Args:
        corpus:{iterable of list of (int, float), scipy.sparse.csc}, )
            Stream of document vectors or sparse matrix of shape (num_documents, num_terms).
            If not given, the model is left untrained (presumably because you want to call update() manually).
        texts:(list of list of str
            Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`) probability estimator .
        dictionary: (Dictionary)
            Gensim dictionary mapping of id word to create corpus. If model.
            id2word is present, this is not needed. If both are provided, passed dictionary will be used.
        k:  Int,
            The number of requested latent topics to be extracted from the training corpus.
        id2word :dict of (int, str) gensim.corpora.dictionary.Dictionary,
            Mapping from word IDs to words.It is used to determine the vocabulary size, as well as for debugging and topic printing.
        alpha:float, numpy.ndarray of float, list of float, str,
            A-priori belief on document-topic distribution, this can be:
               * scalar for a symmetric prior over document-topic distribution,
               * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.
            Alternatively default prior selecting strategies can be employed by supplying a string:
               * ’symmetric’: (default) Uses a fixed symmetric prior of 1.0 / num_topics,
                *’asymmetric’: Uses a fixed normalized asymmetric prior of 1.0 / (topic_index + sqrt(num_topics)).
        eta :float, numpy.ndarray of float, list of float, str,
            A-priori belief on topic-word distribution, this can be:
                *scalar for a symmetric prior over topic-word distribution,
                *1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                *matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.
            Alternatively default prior selecting strategies can be employed by supplying a string:
                *’symmetric’: (default) Uses a fixed symmetric prior of 1.0 / num_topics,
                *’auto’: Learns an asymmetric prior from the corpus.
        path_output: str, default ''
            String with the path to the output folder.

        Returns:
            Coherence value for specific LDA.
            Save current model.

        Raise:
    """
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=k,
        random_state=100,
        # update_every = 1,
        chunksize=100,
        passes=10,
        iterations=100,
        per_word_topics=True,
        alpha=alpha,
        eta=eta,
        workers=lda_cpus,
    )

    lda_model.save(path_output + f"model_LDA{str(k)}_redes.model")
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=text, dictionary=dictionary, coherence="c_v"
    )

    return coherence_model_lda.get_coherence()


def Lda_range(
    df,
    corpus,
    id2word,
    path_output="",
    min_topics=2,
    max_topics=12,
    step_size=1,
    workers=None,
):
    """Model several LDA model and create a table with coherence values for each number of topics (K).

    Iterate over different number of topics (k) to create a table
    with coherence values for each one and give recommended k.
    use compute_coherence_values fo calculation.

    Args:
        df: Pandas or polars dataframe.
        corpus:{iterable of list of (int, float), scipy.sparse.csc}, )
            Stream of document vectors or sparse matrix of shape (num_documents, num_terms).
            If not given, the model is left untrained (presumably because you want to call update() manually)
        id2word :dict of (int, str) gensim.corpora.dictionary.Dictionary,
            Mapping from word IDs to words.It is used to determine the vocabulary size, as well as for debugging and topic printing.
        path_output: str, default ''
            String with the path to the output folder.
        min_topic: int,
            minimum number of topics to evaluate.
        max_topic: int,
            maximum number of topics to evaluate.
        step_size: int,
            number of steps between min_topic and max topic.
    Return:
        recommended  K, int

    Raise:
    """
    model_results = {"Topics": [], "Coherence": []}

    topics_range = range(min_topics, max_topics, step_size)
    persistence = read_or_empty_dict(path_output + "coherence_results.pkl")
    for k in tqdm(topics_range, desc="LDA different k"):
        # get the coherence score for the given parameters
        cv = compute_coherence_values(
            corpus=corpus,
            dictionary=id2word,
            k=k,
            #                                   a=alpha,
            #                                   b=beta,
            text=df["processed"],
        )
        # Save the model results
        persistence[k] = cv
        model_results["Topics"].append(k)
        model_results["Coherence"].append(cv)
    df_m = (
        pd.DataFrame(model_results)
        .sort_values(by="Topics")
        .set_index("Topics")[["Coherence"]]
    )
    x = df_m.index.tolist()
    y = df_m["Coherence"].tolist()
    con_codo = 0
    with open(path_output + "coherence_results.pkl", "wb") as f:
        pickle.dump(persistence, f)
    pd.DataFrame(model_results).to_csv(
        path_output + "coherence_results.csv", index=True
    )
    k = codo_sen(x, y, con_codo)
    return k


def codo_sen(x, y, con_codo, S=1.0):
    """Detect Inflection Point of a Concave Curve (elbow or kneed method for selecting K).

    For a concave curve try to select the inflection point.
    lower the sensitivity for detecting the inflection  to find the inflection. Only 10 times.
    if the inflection point is not found, the maximum is used.

    Args:
        x: list,
            List containing the number of topics.
        y: list,
            List containing the coherence value for each number of topics.

    Raise:

    """

    try:
        if con_codo < 10:
            kneedle = KneeLocator(x, y, S=S, curve="concave", direction="increasing")
            return round(kneedle.elbow, 0)
        else:
            return (
                pd.DataFrame({"coherence": y}, index=x)
                .sort_values(by="coherence", ascending=False)
                .reset_index()
                .rename({"index": "k"})
                .iloc[0]["index"]
            )
    except:
        con_codo += 1
        print("sensitivity to detect elbow not working will be reduced", S)
        return codo_sen(x, y, con_codo, S=S - 0.1)


def write_corpus(corpus, path=""):
    """Saves the corpus

    Saves the corpus in a pickle file.

    Args:
        corpus:{iterable of list of (int, float), scipy.sparse.csc}, )
            Stream of document vectors or sparse matrix of shape (num_documents, num_terms).
            If not given, the model is left untrained (presumably because you want to call update() manually).
    Return:
        no return.


    """
    with open(path + "corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)


def read_corpus(path=""):
    """Read the corpus

    Saves the corpus in a pickle file.

    Args:
        path: str, default ''
            path to the corpus file
    Return:
        corpus:{iterable of list of (int, float), scipy.sparse.csc}, )
            Stream of document vectors or sparse matrix of shape (num_documents, num_terms).
            If not given, the model is left untrained (presumably because you want to call update() manually).

    Raise:

    """

    with open(path + "corpus.pkl", "rb") as f:
        return pickle.load(f)


def lda_Accenture(
    df, min_topics=1, max_topics=12, step_size=1, k=None, path_output="", lda_cpus=None
):
    """Pipe line for implementation for LDA modeling.

    Args:
        df: Pandas or polars dataframe.

        min_topic: int,
            minimum number of topics to evaluate.
        max_topic: int,
            maximum number of topics to evaluate.
            Only used if k is equal to None.
        step_size: int,
            number of steps between min_topic and max topic.
            Only used if k is equal to None.
        k:  Int, default None
            The number of requested latent topics to be extracted from the training corpus.
            If k, only one model with number a of topics equal to k created.

        path_output: str, default ''
            String with the path to the output folder.


    """
    id2word = corpora.Dictionary(df["processed"])
    corpus = [id2word.doc2bow(text) for text in df["processed"]]
    write_corpus(corpus=corpus, path=path_output)

    if k:
        cv = compute_coherence_values(
            corpus=corpus,
            dictionary=id2word,
            k=k,
            id2word=id2word,
            text=df["processed"],
            path_output=path_output,
            lda_cpus=lda_cpus,
        )
    else:
        model_results = {"Topics": [], "Coherence": []}

        topics_range = range(min_topics, max_topics, step_size)
        for k in tqdm(topics_range, desc="LDA different k"):
            # get the coherence score for the given parameters
            cv = compute_coherence_values(
                corpus=corpus,
                dictionary=id2word,
                k=k,
                id2word=id2word,
                text=df["processed"],
                path_output=path_output,
                lda_cpus=lda_cpus,
            )
            # Save the model results

            model_results["Topics"].append(k)
            model_results["Coherence"].append(cv)
        df_m = (
            pd.DataFrame(model_results)
            .sort_values(by="Topics")
            .set_index("Topics")[["Coherence"]]
        )
        df_m.to_csv(path_output + "Coherence_values.csv")
        x = df_m.index.tolist()
        y = df_m["Coherence"].tolist()
        con_codo = 0

        k = codo_sen(x, y, con_codo)
    return k


def write_df(df, format="csv", path_output=""):
    """Save polar or pandas dataframe as csv

    Args:
        df: Pandas or polars dataframe.
        format: str, default ''
            Format of the output file. Only csv is accepted at the time.
        path_output: str, default ''
            String with the path to the output folder.
    Return:
        No return.
        Only write file.

    Raise:
    """
    if format == "csv":
        if isinstance(df, pd.DataFrame):
            df.to_csv(path_output + "df_processed.csv", index=True)
        elif isinstance(df, pl.DataFrame):
            df.write_csv(path_output + "df_processed.csv", has_header=True)


def join_text(df):
    """Join text in a dataframe.

    Args:
        df: Pandas or polars dataframe.
    Return:
        processed dataframe.

    Raise:
    """
    df["processed"] = df["processed"].map(lambda x: " ".join(x))
    return df


def to_str(df):
    """transform the list containin words to str.

    Args:
        df: Pandas or polars dataframe.
    Return:
        processed dataframe.

    Raise:
    """
    df["processed"] = df["processed"].map(lambda x: str(x))
    return df


def from_str(df):
    """the string containing list are transformed into list.

    Args:
        df: Pandas or polars dataframe.
    Return:
        processed dataframe.

    Raise:
    """
    df["processed"] = df["processed"].map(lambda x: eval(x))
    return df


def json_write(filename: str, metadata):
    """Write Json file


    Write a dictionary objet as a json file.

    Args:
        filename: str
            path for the file.
        metadata: dict.
            dict object.
    Return:
        No return.
        Only write file.

    Raise:
    """

    with open(filename, "w") as fp:

        json.dump(metadata, fp)


def json_read(filename: str):
    """Read Json file

    Read Json file.

    Args:
        filename: str
            path of the file.

    Return:
        No return
        Only read file.

    Raise
    """
    with open(filename) as f_in:
        return json.load(f_in)


def write_metadata(
    k_recommended,
    path_data,
    path_output,
    path_output_model,
    lib,
    names,
    filter_list,
    date,
    time_process,
    optimization,
    data_volume,
    n_cpu,
    k,
    min_topics,
    max_topics,
):
    """Recolects metadata in a dictionary and save the info as a JASO file.

    Args:
        k_recommended: int,
            recomende K
        path_data: str,
            path to raw data
        path_output: str,
            path to output data
        path_output_model: str,
            path to LDA Models
        lib: str,
            Used library
        names: dict,
            columns names to change and filter.
        filter_list: List,
            Social network to filter.
        date: str,
            Current date

        time_process: dict,
            dictionary with the time for every process
        optimization:Bool
            Boolean, if true use multiprocess library.
        data_volume: tuple
            data size
        n_cpu: int,
            used number of cpus
        k: int,
            If only one K selected.
        min_topics: int,
            minimum topic to evaluate
        max_topics: int,
            maximum topic to evaluate.


    Return:
     no return.


    """
    metadata = {}
    if k:
        metadata["Range_LDA"] = False
        metadata["Min Topics"] = k
        metadata["Max Topics"] = k
    else:
        metadata["Range_LDA"] = True
        metadata["Min Topics"] = min_topics
        metadata["Max Topics"] = max_topics

    if k_recommended:
        metadata["Recommended k"] = int(k_recommended)
    else:
        metadata["Recommended k"] = "Not K"
    if path_data:
        metadata["Path_raw_data"] = path_data
    else:
        metadata["Path_raw_data"] = "Not path data"
    if path_output:
        metadata["Path_output"] = path_output
    else:
        metadata["Path_output"] = "No path output model"
    if path_output_model:
        metadata["Path_output_model"] = path_output_model
    else:
        metadata["Path_output_model"] = "No path output model"
    if lib:
        metadata["Used Library"] = lib
    else:
        metadata["Used Library"] = "no selected lib"
    if names:
        metadata["Column_used_and_new_names"] = names
    else:
        metadata["Column_used_and_new_names"] = "No selected columns and names"
    if filter_list:
        metadata["Social_networks_used"] = filter_list
    else:
        metadata["Social_networks_used"] = "No social network selection"
    if time_process:
        metadata["Time process"] = time_process
    else:
        metadata["Time process"] = "No Time process"

    metadata["Optimization"] = optimization
    metadata["Number of cpus"] = n_cpu
    metadata["data volume"] = data_volume

    metadata["Start date"] = date
    metadata["End Date"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    # json_write(path_output_model+'metadata.json',metadata)#output of the model
    json_write(path_output + "metadata.json", metadata)  # output of the data


def check_cpus(n_cpus):
    """Check if the number of CPUs is a valid value.

    Check if the number of CPUs provided by the user is valid.
    If the user do not provide value set the maximum number of the machine's CPUs.
    If the user provide a value higher than the maximum number of cpus , set the velue to the maximum number of CPUs -1.
    If the user provide a value lesser tha 1 m set the value to 1.

    Args:
        n_cpus: int,
            number of CPUs to use.

    Return:
        valid number of CPUs




    """

    if not n_cpus:
        n_cpus = cpu_count() - 1
    if n_cpus > cpu_count():
        n_cpus = cpu_count()
        print(
            f"The maximum number of CPUs is {cpu_count()}. Setting the value of n_cpu to {cpu_count()}"
        )
    if n_cpus < 1:
        n_cpus = 1
        print(f"The minimum number of CPUs is 1. Setting the value of n_cpu to 1")
    return n_cpus


def sample_df(df, sample_frac=None, n_volume=None):
    """Function for sampling the dataset.

    Sample the data set without replace.

    Args:
        df: Pandas or polars dataframe.
        sample_frac :float,
            value between 0 and 1.
        n_volume: int,
            number of data to sample. If value for sample_frac is given, only sample_frac is used.
    Return:
        Pandas or polars Sampled dataframe.

    """
    # 42 is the answer to the universe if sample_frac and n_volume , sample_frac is choosen.

    if isinstance(df, pd.DataFrame):
        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42)
        elif n_volume:
            df = df.sample(n=n_volume, random_state=42)
    elif isinstance(df, pl.DataFrame):
        if sample_frac:
            df = df.sample(fraction=sample_frac, seed=42)
        elif n_volume:
            df = df.sample(n=n_volume, seed=42)

    return df


def Social_listening_data_process_and_modeling_LDA(
    sep=",",
    path_data="",
    path_output="",
    path_output_model="",
    lib="polars",
    column_names=[],
    filter_list=[],
    new_stop=[],
    optimization=True,
    min_topics=1,
    max_topics=12,
    step_size=1,
    k=None,
    n_cpu=None,
    n_cpu_lda=None,
    sample_frac=None,
    n_volume=None,
    n_data=None,
):
    """LDA processing pipe line

    Process data with polars or pandas, save models and processed csv. Save metadata of the process.
    Pipeline.
    Steps:
        1 - Check Stopwords Resource.
        2 - Check cpus.
        3 - Get Current Fate
        4 - Filter and change columns Names
        5 - File Reading
        6 - DATA Sample
        7 - Network Distribution
        8 - Drop nan in content
        9 - Repeated Post
        10- Hash table
        11- Drop duplicated
        12- Column dtypes
        13- Filter columns by conten
        14- Text process Tokenization
        15- Drop duplicated ( some publications only are difernt by a name or url) before is require to tranform into string , becouse droping duplicates of lists is more time expensive than for strings.
        16- Bigrams
        17- Lematization and stop words removal process
        18- bigram transformation, Lematization and stop words removal process (many for, requiere optimization)
        19- LDA_modeling
        20- Join words
        21- Write files

    Args:
        sep:str, default ','
            Delimiter to use.
        path: str, required
            Sting with the path to the folder or the file.
        lib: {'pandas', 'polars'} default pandas
            only por pandas or polars
        path_output: str, default ''
            String with the path to the output folder.
        column_name: str, default content
            Name of the column in a dataframe containing text.
        n_cores: int, default cpu_count() maximum number of cpus.
            Number of cores to use. by default is used the maximum of the machine.
        min_topic: int,
            minimum number of topics to evaluate.
        max_topic: int,
            maximum number of topics to evaluate.
            Only used if k is equal to None.
        step_size: int,
            number of steps between min_topic and max topic.
            Only used if k is equal to None.
        k:  Int, default None
            The number of requested latent topics to be extracted from the training corpus.
            If k, only one model with number a of topics equal to k created.
        filter_list: List,
            Social network to filter.
        new_stop: List,
            list of words to add to the stop words list.
        sample_frac :float,
            value between 0 and 1.
        n_volume: int,
            number of data to sample. If va
        n_data: int,
            number of hat to select in order. only for experimentation.

    Return:
        no return. Save model, write csv file with processed data and write metadata.
    """

    print("Start Process LDA")
    time_process = {}
    start_time_ini = time.time()

    ### Check stopwords #################################################################
    check_stopwords_resource()

    #### Check number of CPUs###########################################################
    n_cpu = check_cpus(n_cpu)

    ### Current Date ####################################################################
    date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    ### Filter and change columns Names###################################################
    if not column_names:
        column_names = {
            "Autor": "author",
            "Contenido de la publicación": "content",  # Minimum required
            "Creado": "date",
            "Contexto": "context",
            "Link para la fuente": "link",
            "Grupo de dominio": "domain_group",  # Minimum required
            "País": "country",
        }
    if not filter_list:
        filter_list = ["Twitter", "Facebook", "Instagram", "Photos & Video"]

    ### File Reading #######################################################################
    start_time = time.time()
    print("*Reading")
    df = local_reading(path=path_data, sep=sep, lib=lib)
    data_volume = df.shape
    print(f"Size Of the Data:{data_volume}")
    print("Reading --- %s seconds ---" % (time.time() - start_time))
    time_process["Reading"] = time.time() - start_time

    ### Select rows in Order, only for experimentarion##########################################
    if n_data:
        # take  number of rows in orden , mosstly for experimentation
        df = df[:n_data]
        data_volume = df.shape
        print(f"Size Of the Data Sliced by n_data parameter:{data_volume}")

    ### DATA Sample #############################################################################
    if sample_frac or n_volume:
        df = sample_df(df=df, sample_frac=sample_frac, n_volume=n_volume)
        data_volume = df.shape
        print(
            f"Size Of the Data Sampled by sample_frac or n_volume parameter:{data_volume}"
        )

    ###   Normalize columns ######################################################################
    start_time = time.time()
    print("*Normalize columns")
    df = normalize_columns(df, dir_names=column_names)
    print("Normalize columns --- %s seconds ---" % (time.time() - start_time))
    time_process["Normalize columns"] = time.time() - start_time

    ### Network Distribution #######################################################################
    start_time = time.time()
    print("*Network distribution")
    Network_ditribution(df, path_output)
    print("Network distribution --- %s seconds ---" % (time.time() - start_time))
    time_process["Network distribution"] = time.time() - start_time

    ### Drop nan #################################################################################
    start_time = time.time()
    print("*Drop nan")
    df = drop_nan(df)
    print("Drop nan --- %s seconds ---" % (time.time() - start_time))
    # time_process['Drop nan'] = time.time() - start_time

    ### Repeated Post ###################################################################################
    start_time = time.time()
    print("*Repeated post")
    Repeated_posts(df, path_output=path_output)
    print("Repeated post --- %s seconds ---" % (time.time() - start_time))
    time_process["Repeated post"] = time.time() - start_time

    ### Hash table ####################################################################################
    start_time = time.time()
    print("*Hash tags table")
    hash_table(df, path_output)
    print("Hash tags table --- %s seconds ---" % (time.time() - start_time))
    time_process["Hash tags table"] = time.time() - start_time

    ### Drop duplicated #################################################################################
    start_time = time.time()
    print("*Drop duplicates")
    df = drop_duplicated_nan(df)
    print("Drop duplicates --- %s seconds ---" % (time.time() - start_time))
    data_volume_unique = df.shape
    print(f"Size Of the Data after cleaning:{data_volume_unique}")
    time_process["Drop duplicates"] = time.time() - start_time

    ### Column dtypes ##################################################################################
    start_time = time.time()
    print("*Set data types")
    df = columns_dtypes(df)
    print("Set data types --- %s seconds ---" % (time.time() - start_time))
    time_process["Set data types"] = time.time() - start_time

    ### Filter columns by conten t########################################################################
    start_time = time.time()
    print("*Filter columns of social network")
    df = filter_columns_by_content(df, filter_list, name="domain_group")
    print(
        "Filter columns of social network --- %s seconds ---"
        % (time.time() - start_time)
    )
    time_process["Filter columns of social network"] = time.time() - start_time

    ### Text process Tokenization #########################################################################
    start_time = time.time()
    print("*Text cleaning paralelized")
    if optimization:
        df = parallelize_dataframe(df=df, n_cores=n_cpu, func=denom)
        print(
            "Text cleaning paralelized --- %s seconds ---" % (time.time() - start_time)
        )
        time_process["text cleaning paralelized"] = time.time() - start_time

    else:

        if isinstance(df, pd.DataFrame):
            df = denom(df)
            print("text cleaning  --- %s seconds ---" % (time.time() - start_time))
            time_process["text cleaning"] = time.time() - start_time

        elif isinstance(df, pl.DataFrame):

            df = df.with_columns(
                pl.col("content").apply(sentences_to_words).alias("processed")
            )
            print("text cleaning  --- %s seconds ---" % (time.time() - start_time))
            time_process["text cleaning"] = time.time() - start_time
    ### to string ########################################################################################################
    print("*Transform list into string")
    df = parallelize_dataframe(df=df, func=to_str)
    print("from list to str --- %s seconds ---" % (time.time() - start_time))

    ### Drop duplicated #########################################################################################
    start_time = time.time()
    print("*Drop duplicates after cleaning")
    df = drop_duplicated_nan(df, column="processed")
    print(
        "Drop duplicates after cleaning --- %s seconds ---" % (time.time() - start_time)
    )
    data_volume_unique = df.shape
    print(f"Size Of the Data after cleaning:{data_volume_unique}")
    time_process["Drop duplicates"] = time.time() - start_time

    ### from string to list ########################################################################################################
    print("*From list to string")
    df = parallelize_dataframe(df=df, func=from_str)
    print("from str to list --- %s seconds ---" % (time.time() - start_time))

    ### Bigrams #################################################################################################
    start_time = time.time()
    print("*Bigram instance")
    stop_words = stopwords.words("english")
    stop_words.extend(new_stop)
    bigram = models.Phrases(
        df["processed"], min_count=5, threshold=20
    )  # it need to see all the dataset. for that reason this is outside(not workin for multiprocess)
    bigram_mod = models.phrases.Phraser(bigram)
    print("Bigram instance --- %s seconds ---" % (time.time() - start_time))
    time_process["Bigram instance"] = time.time() - start_time

    ### bigram transformation, Lematization and stop words removal process ##########################################
    start_time = time.time()
    print("*text procesing")
    if optimization:
        df = parallelize_dataframe(
            df=df,
            n_cores=n_cpu,
            func=denom_process_words,
            stop_words=stop_words,
            bigram_mod=bigram_mod,
        )
        print("text procesing --- %s seconds ---" % (time.time() - start_time))
        time_process["text procesing"] = time.time() - start_time

    else:
        ##################WIP###########
        if isinstance(df, pd.DataFrame):
            df = denom_process_words(df, stop_words=stop_words, bigram_mod=bigram_mod)
            print("text procesing  --- %s seconds ---" % (time.time() - start_time))
            time_process["text procesing"] = time.time() - start_time

        elif isinstance(df, pl.DataFrame):
            df = df.to_pandas()
            df = denom_process_words(df, stop_words=stop_words, bigram_mod=bigram_mod)
            df = pl.from_pandas(df)
            print("text procesing  --- %s seconds ---" % (time.time() - start_time))
            time_process["text procesing"] = time.time() - start_time

    ### LDA Model #########################################################################################################
    start_time = time.time()
    print("*Modeling LDA")
    k_recomended = lda_Accenture(
        df=df,
        path_output=path_output_model,
        min_topics=min_topics,
        max_topics=max_topics,
        step_size=step_size,
        k=k,
        lda_cpus=n_cpu_lda,
    )
    print("*Modeling LDA --- %s seconds ---" % (time.time() - start_time))
    time_process["Modeling LDA"] = time.time() - start_time

    start_time = time.time()

    ### Join words ########################################################################################################
    print("*Joing words")
    df = parallelize_dataframe(df=df, func=join_text)
    print("Joing words --- %s seconds ---" % (time.time() - start_time))
    time_process["Joing words"] = time.time() - start_time

    time_process["End of process"] = time.time() - start_time_ini

    ### Write files ###############################################################################################
    write_df(df, path_output=path_output)
    write_metadata(
        k_recommended=k_recomended,
        path_data=path_data,
        path_output=path_output,
        path_output_model=path_output_model,
        lib=lib,
        names=column_names,
        filter_list=filter_list,
        date=date,
        time_process=time_process,
        optimization=optimization,
        data_volume=data_volume,
        n_cpu=n_cpu,
        min_topics=min_topics,
        max_topics=max_topics,
        k=k,
    )

    print(print("End of process --- %s seconds ---" % (time.time() - start_time_ini)))


def load_gensim_lda_model(path_models: str, NUM_TOPICS: int):
    """Read Gensim LDA models

    Read The LDA Models.

    Args:
        path_models: str,
            path to the save model.
        NUM_TOPICS: int,
            number of topics for selecting diferent LDA model.
    Return:
        Gensim LDA model.

    """
    return models.ldamodel.LdaModel.load(
        f"{path_models}model_LDA{str(NUM_TOPICS)}_redes.model"
    )


def lda_publications_classifier(corpus, lda_model):
    """LDA Classifier

    Use the LDA model for text clasification.

    Args:
        corpus: list,
            list with the etiquete for every token (corpus)
        lda_model:  gensim lda model,
            lda model.
    Return: pandas DataFrame,
        pandas dataframe containing the probability to belong to the diferent topics.



    """
    lista_topics = []
    lista_proba = []
    cont = 0
    for frase in corpus:
        cont += 1
        # print(cont)
        topic_percs, wordid_topics, wordid_phivalues = lda_model[frase]
        topic_percs_sorted = sorted(
            topic_percs, key=lambda x: (x[1]), reverse=True
        )  # list of the final clasification

        lista_topics.append(topic_percs_sorted[0][0])
        lista_proba.append(topic_percs)

    df_proba = pd.DataFrame(lista_proba).add_prefix("Topic_").add_suffix("_proba")
    df_proba.insert(0, "Topic", lista_topics)

    return df_proba


def optimized_lda_classifier(corpus, lda_model, n_cpus=None, optimization=True):
    """Optimize the clasification process

    Paralelize the proces of clasification.

    Args:
        corpus: list,
            list with the etiquete for every token (corpus)
        lda_model:  gensim lda model,
            lda model.
        n_cpu: int,
            used number of cpus
        optimization:Bool
            Boolean, if true use multiprocess library.
    Return: pandas DataFrame,
        pandas dataframe containing the probability to belong to the diferent topics.




    """
    if optimization:
        n_cpus = check_cpus(n_cpus)
        pool = Pool(n_cpus)
        df = pd.Series(corpus, name="corpus").to_frame()
        df = np.array_split(df, n_cpus)
        async_results_cl = [
            pool.apply_async(
                lda_publications_classifier, args=(i.loc[:, "corpus"], lda_model)
            )
            for i in df
        ]

        # return list(itertools.chain.from_iterable(x)) for list of list
        return [ar.get() for ar in async_results_cl]

    else:
        return lda_publications_classifier(corpus=corpus, lda_model=lda_model)


def read_or_empty_dict(path):
    """
    Load a pickle file if it exists, or return an empty dictionary if it doesn't.
    Args:
        path (str): Path to the pickle file.
    Returns:
        dict: Loaded dictionary or an empty dictionary.
    """
    if os.path.isfile(path):
        # File exists, load the dictionary
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                # Handle exceptions (e.g., corrupted pickle file)
                pass
    return {}  # Return an empty dictionary if the file doesn't exist
