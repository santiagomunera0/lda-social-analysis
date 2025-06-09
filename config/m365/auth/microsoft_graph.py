# config/m365/auth/microsoft_graph.py
import msal
import requests
import os
#from ..utils.temp_utils import temp_manager

# Token cache file path
TOKEN_CACHE_FILE = os.path.join(os.path.expanduser("~"), ".ms_graph_token_cache.json")

def authenticate(client_id, auth, scopes):
    """
    Autentica al usuario con Microsoft y obtiene un token de acceso válido para Microsoft Graph.
    
    Este método implementa el flujo de autenticación de OAuth 2.0 usando la biblioteca MSAL.
    Primero intenta usar una sesión existente (token en caché) y, si no está disponible,
    inicia el flujo de autenticación interactiva que abrirá un navegador web.
    
    Requiere que las constantes CLIENT_ID, AUTHORITY y GRAPH_SCOPES estén definidas
    en el ámbito global.
    
    Returns:
        dict: Respuesta de token que contiene access_token, refresh_token, expires_in, etc.
            Retorna None si la autenticación falla.
    
    Ejemplo:
        >>> token_response = authenticate()
        >>> if token_response and 'access_token' in token_response:
        >>>     # Usuario autenticado correctamente
        >>>     print(f"Token válido por {token_response['expires_in']} segundos")
    """
    # Create token cache
    cache = msal.SerializableTokenCache()
    
    # Load the token cache from file if it exists
    if os.path.exists(TOKEN_CACHE_FILE):
        with open(TOKEN_CACHE_FILE, 'r') as cache_file:
            cache.deserialize(cache_file.read())
    
    # Create MSAL app with the cache
    app = msal.PublicClientApplication(
        client_id, 
        authority=auth,
        token_cache=cache
    )
    
    # Try to get token silently
    accounts = app.get_accounts()
    result = None
    if accounts:
        print(f"✅ Usando sesión existente para {accounts[0]['username']}")
        result = app.acquire_token_silent(scopes, account=accounts[0])
    
    # If no token or silent acquisition fails, authenticate interactively
    if not result:
        print("🌐 No hay sesión guardada. Abriendo navegador para autenticación...")
        result = app.acquire_token_interactive(scopes)
    
    # Save cache if token was obtained
    if result and 'access_token' in result:
        # Save cache to file
        if cache.has_state_changed:
            with open(TOKEN_CACHE_FILE, 'w') as cache_file:
                cache_file.write(cache.serialize())
    
    return result

def get_headers(token_response):
    """
    Construye los headers de autorización necesarios para las peticiones a Microsoft Graph.
    
    Toma la respuesta del token de autenticación y genera un diccionario con el header
    de autorización en el formato requerido por la API de Microsoft Graph.
    
    Args:
        token_response (dict): Respuesta de token obtenida de la función authenticate().
            Debe contener una clave 'access_token' con el token JWT válido.
    
    Returns:
        dict: Diccionario con el header 'Authorization' formateado correctamente.
            Ejemplo: {"Authorization": "Bearer eyJ0eXAi..."}
            Retorna None si token_response no contiene un access_token.
    
    Ejemplo:
        >>> token_response = authenticate()
        >>> headers = get_headers(token_response)
        >>> if headers:
        >>>     # Usar headers para hacer peticiones a la API
        >>>     response = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers)
    """
    return {"Authorization": f"Bearer {token_response['access_token']}"} if "access_token" in token_response else None


def verify_auth(token_response=None, headers=None):
    """
    Verifica si el usuario está autenticado correctamente con Microsoft Graph.
    
    Esta función puede verificar la autenticación usando dos métodos:
    1. Comprobar si existe un token_response con access_token válido
    2. Validar los headers existentes realizando una petición de prueba a la API
    
    Args:
        token_response (dict, optional): Respuesta de token obtenida de authenticate().
        headers (dict, optional): Headers de autorización para Microsoft Graph.
            Al menos uno de los dos parámetros debe proporcionarse.
    
    Returns:
        bool: True si el usuario está correctamente autenticado, False en caso contrario.
    
    Ejemplo:
        >>> # Verificar con token_response
        >>> token_response = authenticate()
        >>> is_auth = verify_auth(token_response=token_response)
        >>> 
        >>> # O verificar con headers
        >>> headers = get_headers(token_response)
        >>> is_auth = verify_auth(headers=headers)
    """
    if token_response and "access_token" in token_response:
        return True
    elif headers and "Authorization" in headers:
        # Opcionalmente puedes validar que el token sea válido haciendo una petición simple
        test_response = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers)
        return test_response.status_code == 200
    return False


def require_auth(func):
    """
    Decorador que verifica la autenticación antes de ejecutar una función.
    
    Este decorador está diseñado para ser utilizado con funciones que requieren
    autenticación con Microsoft Graph. Verifica automáticamente si los headers
    proporcionados contienen un token válido antes de ejecutar la función decorada.
    Si la autenticación falla, muestra un mensaje y devuelve None sin ejecutar la función.
    
    Args:
        func (callable): La función a decorar. Esta función debe tener 'headers'
            como su primer parámetro.
    
    Returns:
        callable: Una función wrapper que verifica la autenticación antes de llamar a func.
    
    Ejemplo:
        >>> @require_auth
        >>> def get_user_profile(headers):
        >>>     response = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers)
        >>>     return response.json()
        >>> 
        >>> # La función ahora verificará automáticamente la autenticación
        >>> profile = get_user_profile(headers)
    """
    def wrapper(headers, *args, **kwargs):
        if not verify_auth(headers=headers):
            print("❌ Usuario no autenticado o token expirado.")
            return None
        return func(headers, *args, **kwargs)
    return wrapper


@require_auth
def get_user_info(headers):
    """
    Obtiene la información del perfil del usuario actualmente autenticado.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
            Ejemplo: {"Authorization": "Bearer eyJ0eXAi..."}
    
    Returns:
        dict: Datos del perfil del usuario con campos como displayName, mail, userPrincipalName, etc.
            Retorna None si la solicitud falla o el usuario no está autenticado.
    
    Ejemplo:
        >>> headers = get_headers(token_response)
        >>> user_info = get_user_info(headers)
        >>> print(f"Usuario: {user_info['displayName']}")
    """
    response = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers)
    return response.json() if response.status_code == 200 else None

@require_auth
def get_joined_teams(headers):
    """
    Obtiene la lista completa de equipos de Microsoft Teams a los que pertenece el usuario.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
            Ejemplo: {"Authorization": "Bearer eyJ0eXAi..."}
    
    Returns:
        list: Lista de equipos con sus propiedades (id, displayName, description, etc.).
            Retorna una lista vacía si no hay equipos o None si la solicitud falla.
    
    Ejemplo:
        >>> headers = get_headers(token_response)
        >>> teams = get_joined_teams(headers)
        >>> for team in teams:
        >>>     print(f"Equipo: {team['displayName']}")
    """
    response = requests.get("https://graph.microsoft.com/v1.0/me/joinedTeams", headers=headers)
    return response.json().get("value", []) if response.status_code == 200 else None

@require_auth
def get_team_id(headers, dir_cliente):
    """
    Busca y obtiene el ID de un equipo de Microsoft Teams que contenga el nombre del cliente.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
        dir_cliente (str): Texto a buscar en el nombre del equipo. La búsqueda no distingue
            entre mayúsculas y minúsculas.
    
    Returns:
        str: ID del primer equipo que contenga el texto especificado en su nombre.
            Retorna None si no se encuentra ningún equipo que coincida.
    
    Ejemplo:
        >>> headers = get_headers(token_response)
        >>> team_id = get_team_id(headers, "Empresa ABC")
        >>> if team_id:
        >>>     print(f"ID del equipo encontrado: {team_id}")
    """
    teams = get_joined_teams(headers)
    if teams:
        for team in teams:
            if dir_cliente.lower() in team["displayName"].lower():
                return team["id"]
    return None

@require_auth
def get_sharepoint_site(headers, team_id):
    """
    Obtiene la información del sitio de SharePoint asociado a un equipo de Microsoft Teams.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
        team_id (str): ID del equipo de Microsoft Teams del cual se quiere obtener el sitio.
    
    Returns:
        dict: Datos del sitio de SharePoint con campos como id, name, webUrl, etc.
            Retorna None si la solicitud falla o el sitio no existe.
    
    Ejemplo:
        >>> headers = get_headers(token_response)
        >>> team_id = get_team_id(headers, "Empresa ABC")
        >>> site = get_sharepoint_site(headers, team_id)
        >>> if site:
        >>>     print(f"URL del sitio: {site['webUrl']}")
    """
    response = requests.get(f"https://graph.microsoft.com/v1.0/groups/{team_id}/sites/root", headers=headers)
    return response.json() if response.status_code == 200 else None

@require_auth
def get_drive_items(headers, site_id):
    """
    Obtiene la lista de elementos (archivos y carpetas) en la raíz del drive de un sitio de SharePoint.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
        site_id (str): ID del sitio de SharePoint del cual se quieren obtener los elementos.
            Normalmente en el formato: "dominio.sharepoint.com,GUID,GUID"
    
    Returns:
        list: Lista de elementos con sus propiedades (id, name, webUrl, folder, file, etc.).
            Retorna una lista vacía si no hay elementos o None si la solicitud falla.
    
    Ejemplo:
        >>> headers = get_headers(token_response)
        >>> site = get_sharepoint_site(headers, team_id)
        >>> items = get_drive_items(headers, site['id'])
        >>> for item in items:
        >>>     print(f"Nombre: {item['name']}, Tipo: {'Carpeta' if 'folder' in item else 'Archivo'}")
    """
    response = requests.get(f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root/children", headers=headers)
    return response.json().get("value", []) if response.status_code == 200 else None

@require_auth
def get_folder_contents(headers, site_id, folder_id):
    """
    Obtiene la lista de elementos (archivos y carpetas) dentro de una carpeta específica en SharePoint.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
        site_id (str): ID del sitio de SharePoint donde se encuentra la carpeta.
            Normalmente en el formato: "dominio.sharepoint.com,GUID,GUID"
        folder_id (str): ID de la carpeta de la cual se quieren obtener los contenidos.
    
    Returns:
        list: Lista de elementos con sus propiedades (id, name, webUrl, folder, file, etc.).
            Retorna una lista vacía si la carpeta está vacía o None si la solicitud falla.
    
    Ejemplo:
        >>> headers = get_headers(token_response)
        >>> site = get_sharepoint_site(headers, team_id)
        >>> items = get_drive_items(headers, site['id'])
        >>> folder = next((item for item in items if 'folder' in item and item['name'] == 'Documentos'), None)
        >>> if folder:
        >>>     contents = get_folder_contents(headers, site['id'], folder['id'])
        >>>     for item in contents:
        >>>         print(f"Elemento: {item['name']}")
    """
    response = requests.get(f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{folder_id}/children", headers=headers)
    return response.json().get("value", []) if response.status_code == 200 else None

@require_auth
def create_folder_path(headers, site_id, parent_folder_id, path, separator="/"):
    """
    Crea una estructura de carpetas a partir de una ruta.
    
    Args:
        headers: Headers de autorización
        site_id: ID del sitio de SharePoint
        parent_folder_id: ID de la carpeta padre donde comenzará la creación
        path: Ruta de carpetas (ej: "Proyecto/2023/v1" o "Proyecto/2023/v1/data")
        separator: Separador utilizado en la ruta (por defecto '/')
    
    Returns:
        ID de la última carpeta creada o None si hay error
    """
    # Si la ruta está vacía, simplemente devolver el ID de la carpeta padre
    if not path:
        return parent_folder_id
    
    # Dividir la ruta en componentes de carpeta
    folder_structure = [folder for folder in path.split(separator) if folder]
    
    if not folder_structure:
        return parent_folder_id
    
    # Comenzamos desde la carpeta padre especificada
    current_parent_id = parent_folder_id
    
    for folder_name in folder_structure:
        # Verificar si la carpeta ya existe
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{current_parent_id}/children"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Error al verificar existencia de la carpeta '{folder_name}': {response.text}")
            return None
        
        existing_folders = response.json().get("value", [])
        folder = next((f for f in existing_folders if f["name"].lower() == folder_name.lower()), None)

        if folder:
            print(f"📂 La carpeta '{folder_name}' ya existe.")
            current_parent_id = folder["id"]
        else:
            # Crear la nueva carpeta
            data = {
                "name": folder_name,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "rename"
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                folder = response.json()
                print(f"✅ Carpeta '{folder_name}' creada exitosamente.")
                current_parent_id = folder["id"]
            else:
                print(f"❌ Error al crear la carpeta '{folder_name}': {response.text}")
                return None

    return current_parent_id


@require_auth
def get_folder_id(headers, site_id, parent_folder_id, path, separator="/"):
    """
    Obtiene el ID de una carpeta en SharePoint navegando por la estructura de carpetas.

    Args:
        headers (dict): Headers de autorización.
        site_id (str): ID del sitio de SharePoint.
        parent_folder_id (str): ID de la carpeta raíz desde donde empezar la búsqueda.
        path (str): Ruta de la carpeta (ejemplo: "Desarrollo/prueba/2025-01-01 2025-01-31/v1.0").
        separator (str): Separador usado en la ruta (por defecto "/").

    Returns:
        str: ID de la carpeta final si existe, o None si no existe.
    """
    folder_names = [folder.strip() for folder in path.split(separator) if folder]
    current_parent_id = parent_folder_id

    for folder_name in folder_names:
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{current_parent_id}/children"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"❌ Error al acceder a la carpeta '{folder_name}': {response.text}")
            return None

        folders = response.json().get("value", [])
        folder = next((f for f in folders if f["name"].lower() == folder_name.lower()), None)

        if folder:
            current_parent_id = folder["id"]
        else:
            print(f"⚠️ La carpeta '{folder_name}' no existe en la ruta '{path}'.")
            return None

    return current_parent_id

@require_auth
def create_folder(headers, site_id, parent_folder_id, path, separator="/"):
    """
    Crea una estructura de carpetas en SharePoint sin necesidad de verificar previamente si existen.
    Solo verifica si hay un error al intentar crearlas.

    Args:
        headers (dict): Headers de autorización.
        site_id (str): ID del sitio de SharePoint.
        parent_folder_id (str): ID de la carpeta raíz desde donde empezar.
        path (str): Ruta de la carpeta a crear.
        separator (str): Separador usado en la ruta (por defecto "/").

    Returns:
        str: ID de la última carpeta creada/existente.
    """
    folder_names = [folder.strip() for folder in path.split(separator) if folder]
    current_parent_id = parent_folder_id

    for folder_name in folder_names:
        print(f"📁 Intentando crear '{folder_name}'...")

        # Intentar crear la carpeta sin verificar si existe
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{current_parent_id}/children"
        data = {
            "name": folder_name,
            "folder": {},
            "@microsoft.graph.conflictBehavior": "fail"  # Evita sobrescribir si ya existe
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 201:
            folder = response.json()
            print(f"✅ Carpeta '{folder_name}' creada exitosamente.")
            current_parent_id = folder["id"]
        elif response.status_code == 409:  # Código 409: Conflicto (la carpeta ya existe)
            print(f"📂 La carpeta '{folder_name}' ya existe.")
            
            # Obtener la carpeta existente sin hacer otra consulta innecesaria
            response_get = requests.get(url, headers=headers)
            if response_get.status_code == 200:
                folders = response_get.json().get("value", [])
                folder = next((f for f in folders if f["name"].lower() == folder_name.lower()), None)
                if folder:
                    current_parent_id = folder["id"]
            else:
                print(f"⚠️ No se pudo obtener el ID de la carpeta existente '{folder_name}'.")
                return None
        else:
            print(f"❌ Error al crear la carpeta '{folder_name}': {response.text}")
            return None

    return current_parent_id

# @require_auth
# def download_file(headers, site_id, file_id, local_path=None, create_missing_dirs=True, get_filename=False):
#     """
#     Descarga un archivo desde SharePoint (Teams) y lo guarda en el sistema local.
    
#     Args:
#         headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
#         site_id (str): ID del sitio de SharePoint donde se encuentra el archivo.
#         file_id (str): ID del archivo a descargar.
#         local_path (str, optional): Ruta local donde se guardará el archivo.
#             Si no se proporciona, se guardará en una ubicación temporal.
#             Si es un directorio, se conservará el nombre original del archivo.
#             Si es una ruta completa, se usará ese nombre de archivo.
#         create_missing_dirs (bool, optional): Si es True, crea los directorios necesarios
#             en la ruta local si no existen. Por defecto es True.
#         get_filename (bool, optional): Si es True, primero obtiene los metadatos del archivo
#             para conocer su nombre original. Por defecto es False para optimizar rendimiento
#             cuando ya se conoce la ruta completa de destino.
    
#     Returns:
#         str: Ruta local del archivo descargado o None si la descarga falla.
#     """
#     file_name = None
    
#     # Si se requiere conocer el nombre del archivo o local_path es un directorio
#     if get_filename or (local_path and os.path.isdir(local_path)):
#         # Obtener metadatos del archivo para saber su nombre
#         url_metadata = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{file_id}"
#         metadata_response = requests.get(url_metadata, headers=headers)
        
#         if metadata_response.status_code != 200:
#             print(f"❌ Error al obtener metadatos del archivo: {metadata_response.text}")
#             return None
            
#         file_metadata = metadata_response.json()
#         file_name = file_metadata.get("name")
        
#         # Si local_path es un directorio, añadir el nombre del archivo
#         if local_path and os.path.isdir(local_path):
#             local_path = os.path.join(local_path, file_name)
    
#     # URL de la API para obtener el contenido del archivo
#     url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{file_id}/content"
#     response = requests.get(url, headers=headers, stream=True)
    
#     if response.status_code == 200:
#         # Determinar la ruta donde se guardará el archivo
#         #if not local_path:
#         #    local_path = temp_manager.get_temp_path(file_id)  # Usa la carpeta temporal gestionada
        
#         # Verificar si es necesario crear directorios
#         directory = os.path.dirname(local_path)
#         if directory and not os.path.exists(directory):
#             if create_missing_dirs:
#                 try:
#                     os.makedirs(directory)
#                     print(f"✅ Directorio creado: {directory}")
#                 except Exception as e:
#                     print(f"❌ Error al crear directorio '{directory}': {str(e)}")
#                     return None
#             else:
#                 print(f"❌ El directorio '{directory}' no existe y create_missing_dirs=False.")
#                 return None
        
#         # Guardar el archivo localmente
#         try:
#             with open(local_path, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
            
#             print(f"✅ Archivo descargado en: {local_path}")
#             return local_path
#         except Exception as e:
#             print(f"❌ Error al guardar el archivo: {str(e)}")
#             return None
#     else:
#         print(f"❌ Error al descargar el archivo: {response.text}")
#         return None

@require_auth
def download_file(headers, site_id, file_id, local_path, create_missing_dirs=True):
    """
    Descarga un archivo específico desde SharePoint.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
        site_id (str): ID del sitio de SharePoint donde se encuentra el archivo.
        file_id (str): ID del archivo a descargar.
        local_path (str): Ruta local donde se guardará el archivo descargado.
            Si es un directorio, se usará el nombre del archivo obtenido de SharePoint.
            Si es una ruta completa, se usará tal cual.
        create_missing_dirs (bool, optional): Si es True, crea los directorios locales si no existen.
            Por defecto es True.
    
    Returns:
        str: Ruta local del archivo descargado si la descarga fue exitosa, None en caso contrario.
    """
    # Verificar si el directorio local existe, si no, crearlo si create_missing_dirs es True
    if not os.path.exists(local_path):
        if create_missing_dirs:
            try:
                os.makedirs(local_path)
                print(f"✅ Directorio local creado: {local_path}")
            except Exception as e:
                print(f"❌ Error al crear directorio local '{local_path}': {str(e)}")
                return None
        else:
            print(f"❌ El directorio local '{local_path}' no existe.")
            return None

    # Obtener los metadatos del archivo para conocer su nombre
    url_metadata = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{file_id}"
    metadata_response = requests.get(url_metadata, headers=headers)
    
    if metadata_response.status_code != 200:
        print(f"❌ Error al obtener metadatos del archivo: {metadata_response.text}")
        return None
    
    file_metadata = metadata_response.json()
    file_name = file_metadata.get("name")
    
    # Determinar la ruta final del archivo
    if os.path.isdir(local_path):
        item_file_path = os.path.join(local_path, file_name)  # Si es un directorio, combinar con el nombre del archivo
    else:
        item_file_path = local_path  # Si es una ruta completa, usarla tal cual
    
    # Descargar el contenido del archivo
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{file_id}/content"
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 200:
        try:
            with open(item_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filtrar chunks vacíos
                        f.write(chunk)
            
            print(f"✅ Archivo descargado exitosamente en: {item_file_path}")
            return item_file_path
        except Exception as e:
            print(f"❌ Error al guardar el archivo '{item_file_path}': {str(e)}")
            return None
    else:
        print(f"❌ Error al descargar el archivo: {response.text}")
        return None

@require_auth
def upload_file(headers, site_id, folder_id, file_path, file_name, conflict_behavior="fail"):
    """
    Uploads a file to a specific folder in SharePoint.

    Args:
        headers (dict): Headers of authorization including the access token.
        site_id (str): ID of the SharePoint site.
        folder_id (str): ID of the folder where the file will be uploaded.
        file_path (str): Local path to the file to be uploaded.
        file_name (str): Name of the file to be uploaded.
        conflict_behavior (str): How to handle conflicts. Options:
            - "replace" (default): Overwrite existing file
            - "rename": Create new file with unique name
            - "fail": Fail if file exists

    Returns:
        dict: Response data if successful or None if failed
    """
    # Validate conflict behavior parameter
    valid_behaviors = ["replace", "rename", "fail"]
    if conflict_behavior not in valid_behaviors:
        print(f"❌ Invalid conflict_behavior: '{conflict_behavior}'. Must be one of {valid_behaviors}")
        return None
    
    # Construct the upload URL with conflict behavior parameter
    upload_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{folder_id}:/{file_name}:/content"
    upload_url += f"?@microsoft.graph.conflictBehavior={conflict_behavior}"
    
    # Set the headers for the upload request
    upload_headers = {
        "Authorization": headers["Authorization"],
        "Content-Type": "application/octet-stream"
    }
    
    try:
        # Read the file content
        with open(file_path, "rb") as file:
            file_content = file.read()
        
        # Perform the upload request
        upload_response = requests.put(upload_url, headers=upload_headers, data=file_content)
        
        # Check the response status
        if upload_response.status_code in [200, 201]:
            response_data = upload_response.json()
            print(f"✅ File '{file_name}' uploaded successfully.")
            if conflict_behavior == "rename" and file_name != response_data.get("name"):
                print(f"ℹ️ File renamed to '{response_data.get('name')}' due to conflict.")
            return response_data
        else:
            error_msg = upload_response.json().get("error", {}).get("message", upload_response.text)
            print(f"❌ Error uploading file '{file_name}': {error_msg}")
            return None
    except Exception as e:
        print(f"❌ Exception uploading file '{file_name}': {str(e)}")
        return None

@require_auth
def upload_folder(headers, site_id, parent_folder_id, local_folder_path, remote_folder_path, 
                  separator="/", conflict_behavior="fail", skip_on_error=False):
    """
    Sube una carpeta completa (con todos sus archivos y subcarpetas) a SharePoint.

    Args:
        headers (dict): Headers de autorización.
        site_id (str): ID del sitio de SharePoint.
        parent_folder_id (str): ID de la carpeta padre en SharePoint.
        local_folder_path (str): Ruta local de la carpeta que se va a subir.
        remote_folder_path (str): Ruta de la carpeta en SharePoint (ej: "dir_escucha/dir_periodo/dir_version/output/topologia").
        separator (str): Separador usado en la ruta (por defecto "/").
        conflict_behavior (str): Comportamiento en caso de conflicto (replace, rename, fail).
        skip_on_error (bool): Si es True, continúa el proceso aunque algún archivo falle.

    Returns:
        dict: Resultados del proceso de subida con estadísticas.
    """
    results = {
        "success": True,
        "files_uploaded": 0,
        "files_failed": 0,
        "folders_created": 0,
        "errors": []
    }
    
    # Obtener el ID de la carpeta en SharePoint, si no existe, crearla
    remote_folder_id = get_folder_id(headers, site_id, parent_folder_id, remote_folder_path, separator)
    if not remote_folder_id:
        remote_folder_id = create_folder_path(headers, site_id, parent_folder_id, remote_folder_path, separator)
        if not remote_folder_id:
            error_msg = f"No se pudo crear la carpeta '{remote_folder_path}' en SharePoint."
            print(f"❌ {error_msg}")
            results["success"] = False
            results["errors"].append(error_msg)
            return results
        results["folders_created"] += 1

    print(f"📂 Carpeta '{remote_folder_path}' encontrada o creada en SharePoint con ID: {remote_folder_id}")

    # Verificar si la carpeta local existe
    if not os.path.exists(local_folder_path):
        error_msg = f"La carpeta local '{local_folder_path}' no existe."
        print(f"❌ {error_msg}")
        results["success"] = False
        results["errors"].append(error_msg)
        return results

    # Recorrer todos los archivos y subcarpetas en la carpeta local
    for item_name in os.listdir(local_folder_path):
        item_path = os.path.join(local_folder_path, item_name)

        if os.path.isfile(item_path):
            # Subir archivo
            upload_result = upload_file(headers, site_id, remote_folder_id, item_path, item_name, conflict_behavior)
            if upload_result:
                results["files_uploaded"] += 1
                print(f"✅ Archivo '{item_name}' subido correctamente.")
            else:
                results["files_failed"] += 1
                error_msg = f"Error al subir el archivo '{item_name}'."
                results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
                if not skip_on_error:
                    results["success"] = False
                    return results
        
        elif os.path.isdir(item_path):
            # Subir subcarpeta recursivamente
            remote_subfolder_path = f"{remote_folder_path}/{item_name}" if remote_folder_path else item_name
            subfolder_results = upload_folder(
                headers, site_id, parent_folder_id, item_path, remote_subfolder_path,
                separator, conflict_behavior, skip_on_error
            )
            
            # Actualizar resultados con los de la subcarpeta
            results["files_uploaded"] += subfolder_results["files_uploaded"]
            results["files_failed"] += subfolder_results["files_failed"]
            results["folders_created"] += subfolder_results["folders_created"]
            results["errors"].extend(subfolder_results["errors"])
            
            if not subfolder_results["success"] and not skip_on_error:
                results["success"] = False
                return results

    return results

@require_auth
def download_folder(headers, site_id, folder_id, local_path, create_missing_dirs=True):
    """
    Descarga una carpeta completa (con todos sus archivos y subcarpetas) desde SharePoint.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
        site_id (str): ID del sitio de SharePoint donde se encuentra la carpeta.
        folder_id (str): ID de la carpeta a descargar.
        local_path (str): Ruta local donde se guardará la carpeta descargada.
        create_missing_dirs (bool, optional): Si es True, crea los directorios locales si no existen.
            Por defecto es True.
    
    Returns:
        bool: True si la carpeta y todos sus archivos se descargaron correctamente, False en caso contrario.
    """
    # Verificar si el directorio local existe, si no, crearlo si create_missing_dirs es True
    if not os.path.exists(local_path):
        if create_missing_dirs:
            try:
                os.makedirs(local_path)
                print(f"✅ Directorio local creado: {local_path}")
            except Exception as e:
                print(f"❌ Error al crear directorio local '{local_path}': {str(e)}")
                return False
        else:
            print(f"❌ El directorio local '{local_path}' no existe.")
            return False
            
    # Obtener los contenidos de la carpeta en SharePoint
    contents = get_folder_contents(headers, site_id, folder_id)
    if contents is None:
        print(f"❌ Error al obtener contenidos de la carpeta en SharePoint.")
        return False
        
    if not contents:
        print(f"ℹ️ La carpeta en SharePoint está vacía.")
        return True
        
    # Procesar cada elemento en la carpeta
    for item in contents:
        item_name = item["name"]
        
        if "folder" in item:
            
            item_local_path = os.path.join(local_path, item_name)
            # Es una subcarpeta, descargar recursivamente
            print(f"📂 Descargando subcarpeta: {item_name}")
            subfolder_success = download_folder(
                headers, 
                site_id, 
                item["id"], 
                item_local_path,  # Usar item_local_path como la ruta de la subcarpeta
                create_missing_dirs
            )
            if not subfolder_success:
                print(f"❌ Error al descargar la subcarpeta '{item_name}'.")
                return False
        elif "file" in item:
            item_local_path = os.path.join(local_path)
            # Es un archivo, descargarlo
            print(f"📄 Descargando archivo: {item_name}")
            file_path = download_file(headers, site_id, item["id"], item_local_path, create_missing_dirs=False)
            if not file_path:
                print(f"❌ Error al descargar el archivo '{item_name}'.")
                return False
                
    print(f"✅ Carpeta descargada exitosamente en: {local_path}")
    return True

@require_auth
def download_folder_by_path(headers, site_id, parent_folder_id, remote_folder_path, local_path, separator="/", create_missing_dirs=True):
    """
    Descarga una carpeta completa desde SharePoint usando su ruta en lugar de su ID.
    
    Args:
        headers (dict): Diccionario con los headers de autenticación incluyendo el token de acceso.
        site_id (str): ID del sitio de SharePoint donde se encuentra la carpeta.
        parent_folder_id (str): ID de la carpeta raíz desde donde empezar la búsqueda.
        remote_folder_path (str): Ruta de la carpeta en SharePoint (ej: "Documentos/Proyecto").
        local_path (str): Ruta local donde se guardará la carpeta descargada.
        separator (str): Separador usado en la ruta (por defecto "/").
        create_missing_dirs (bool, optional): Si es True, crea los directorios locales si no existen.
            Por defecto es True.
            
    Returns:
        bool: True si la carpeta y todos sus archivos se descargaron correctamente, False en caso contrario.
    
    Ejemplo:
        >>> headers = get_headers(token_response)
        >>> site = get_sharepoint_site(headers, team_id)
        >>> success = download_folder_by_path(
        >>>     headers, 
        >>>     site['id'], 
        >>>     site['drive']['root']['id'], 
        >>>     "Documentos/Proyecto",
        >>>     "C:/Mi_Proyecto_Local"
        >>> )
    """
    # Obtener el ID de la carpeta en SharePoint usando la ruta
    folder_id = get_folder_id(headers, site_id, parent_folder_id, remote_folder_path, separator)
    if not folder_id:
        print(f"❌ No se pudo encontrar la carpeta '{remote_folder_path}' en SharePoint.")
        return False
        
    # Usar la función principal para descargar la carpeta
    return download_folder(headers, site_id, folder_id, local_path, create_missing_dirs)