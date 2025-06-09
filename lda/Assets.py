import os
import polars as pl
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            logger.info(f"The folder {path_folder} has been created.")
        else:
            logger.info(f"The folder {path_folder} already exists.")
    except OSError as e:
        logger.error(f"Error creating folder {path_folder}: {str(e)}")
        raise

def local_reading(path, format='csv', sep=',', lib='pandas'):
    """Read files.
    
    Read one or several comma-separated values (csv) 
    file into DataFrame with polars or pandas dataframe.

    Args:
      path: str, required
        String with the path to the folder or the file.
      format: str, default csv
        If the path is a folder only reads files with the desired format.
      sep: str, default ','
         Delimiter to use.
      lib: {'pandas', 'polars'} default pandas
        Only for pandas or polars

    Returns:
      A Dataframe object (pandas or polars) 

    Raises:
      IOError: An error occurred accessing the path.
    """
    if lib not in ['pandas', 'polars']:
        raise ValueError("lib must be 'pandas' or 'polars'")
    
    try:
        if os.path.isfile(path):
            if lib == 'polars':
                file = pl.read_csv(path, separator=sep, infer_schema_length=0)
                return file
            else:  # pandas
                file = pd.read_csv(path, sep=sep)
                return file
        
        elif os.path.isdir(path):
            files = [i for i in os.listdir(path) if i.endswith(format)]
            if not files:
                raise IOError(f"No {format} files found in directory {path}")
            
            if lib == 'polars':
                # Transform empty str to null
                dfs = []
                for file_name in tqdm(files, leave=True, desc='Data Reading'):
                    df = pl.read_csv(
                        os.path.join(path, file_name), 
                        separator=sep, 
                        infer_schema_length=0
                    ).with_columns([
                        pl.when(pl.col(pl.Utf8).str.len_chars() == 0)
                        .then(None)
                        .otherwise(pl.col(pl.Utf8))
                        .keep_name()
                    ])
                    dfs.append(df)
                return pl.concat(dfs, how="vertical_relaxed")
            
            else:  # pandas
                dfs = []
                for file_name in tqdm(files, leave=True, desc='Data Reading'):
                    df = pd.read_csv(os.path.join(path, file_name), sep=sep)
                    dfs.append(df)
                return pd.concat(dfs, ignore_index=True)
        
        else:
            raise FileNotFoundError(f"Path not found: {path}")
            
    except Exception as e:
        logger.error(f"Error reading files from {path}: {str(e)}")
        raise