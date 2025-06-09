# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv
# import config.m365.utils.temp_utils as tempu  # Assuming you have tempu installed

# Get project root directory
current_dir = Path().resolve()
project_root = current_dir.parent  # Adjust this based on your structure

# Load .env file
env_path = project_root / ".env"

if not env_path.exists():
    raise FileNotFoundError(f"No se encontró .env en: {env_path}")

load_dotenv(env_path, override=True)

# Get environment variables
dir_cliente = os.getenv("CLIENTE")
dir_escucha = os.getenv("ESCUCHA")
dir_periodo = os.getenv("PERIODO")
dir_version = os.getenv("VERSION")
id_tenant = os.getenv("TENANT_ID")
id_client = os.getenv("CLIENT_ID")
root_path = os.getenv("ROOT_PATH")

# Authentication settings
AUTH_SCOPES = ["openid", "offline_access"]
GRAPH_SCOPES = [
    "https://graph.microsoft.com/Group.ReadWrite.All",
    "https://graph.microsoft.com/Directory.Read.All"
]
AUTHORITY = f"https://login.microsoftonline.com/{id_tenant}"

def ensure_directory_exists(directory_path):
    """
    Verifica si el directorio existe. Si no existe, lo crea.
    Si hay subdirectorios que no existen, solo crea los necesarios.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directorio creado: {directory_path}")
    else:
        print(f"El directorio ya existe: {directory_path}")

# Temporary directory
temp_dir_path = os.path.join(root_path,dir_cliente,dir_escucha,dir_periodo,dir_version)

ensure_directory_exists(temp_dir_path)