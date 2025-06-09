# config/paths.py
import os
from config.settings import dir_escucha, dir_periodo, dir_version, temp_dir_path  # Import from settings

paths = {
    'local': {
        'data': {
            'base': os.path.join(temp_dir_path, "data")
            },
        'output': {
            'base': os.path.join(temp_dir_path, "output"),
            'llm': {
                'base' : os.path.join(temp_dir_path, "llm"),
                'general' : os.path.join(temp_dir_path, "llm", "general"),
                'zoom' : os.path.join(temp_dir_path, "llm", "zoom")
            },
            'lda': {
                'base' : os.path.join(temp_dir_path, "lda"),
                'models' : os.path.join(temp_dir_path, "lda", "models"),
                'figures' : os.path.join(temp_dir_path, "lda", "figures")
            },
            'topologia': {
                'base' : os.path.join(temp_dir_path, "topologia")
            },
            'presentacion': {
                'base' : os.path.join(temp_dir_path, "presentacion"),
                'recursos' : os.path.join(temp_dir_path, "presentacion", "recursos")
            }
        },
        'templates': os.path.join(os.path.abspath("."), 'templates')
    },
    'cloud': {
        'data': {
            'base': f"{dir_escucha}/{dir_periodo}/{dir_version}/data"
            },
        'output': {
            'base': f"{dir_escucha}/{dir_periodo}/{dir_version}/output",
            'llm': {
                'base' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/llm",
                'general' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/llm/general",
                'zoom' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/llm/zoom"
            },
            'lda': {
                'base' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/lda",
                'models' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/lda/models",
                'figures' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/lda/figures"
            },
            'topologia': {
                'base' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/topologia"
            },
            'presentacion': {
                'base' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/presentacion",
                'recursos' : f"{dir_escucha}/{dir_periodo}/{dir_version}/output/presentacion/recursos"
            
            }
        }   
    }
}
