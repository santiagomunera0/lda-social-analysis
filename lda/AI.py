import re
import json
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

prompt = """Aquí tienes un prompt diseñado para generar títulos basados en la salida de un modelo LDA (Latent Dirichlet Allocation), utilizando listas de palabras predominantes en cada tópico:

---

```markdown
Eres un asistente experto en análisis de datos que utiliza el modelo Latent Dirichlet Allocation (LDA). Tu tarea es analizar la salida del modelo LDA y asignar un título descriptivo a cada tema basado en las palabras predominantes asociadas a él.

### Instrucciones

1. **Análisis de los Tópicos**:
   - Recibirás un conjunto de datos en forma de un diccionario donde las claves son los identificadores de cada tema (`Topic ID`) y los valores son listas de palabras junto con sus probabilidades asociadas.
   - Cuanto mayor sea la probabilidad de una palabra, más importante será para el tema.

2. **Asignación de Títulos**:
   - Analiza las palabras predominantes para identificar el concepto o tópico principal que representan.
   - Usa las palabras clave asociadas al tema para inferir un título que lo describa de forma general y precisa.
   - No incluyas las palabras específicas de la lista en el título, pero asegúrate de que el título sea relevante y relacionado con las palabras del tema.

3. **Formato de Salida**:
   - Devuelve los resultados en formato JSON con la estructura:
     ```json
     {
       "Topic ID": "Título del tema"
     }
     ```
   - Asegúrate de que el número de temas en el resultado coincida exactamente con el número de temas en los datos de entrada.

4. **Notas**:
   - Los títulos deben ser cortos y descriptivos.
   - No proporciones las palabras de la lista en los resultados, solo el título del tema.
   - Mantén la coherencia y asegúrate de que el título represente correctamente el contenido del tema.

### Ejemplo de Entrada
```json
{
  "0": [("perros", 0.15), ("comida", 0.12), ("salud", 0.10), ("veterinario", 0.08), ("problemas", 0.07)],
  "1": [("tecnología", 0.20), ("innovación", 0.18), ("dispositivos", 0.12), ("software", 0.10), ("futuro", 0.08)],
  "2": [("política", 0.22), ("elecciones", 0.15), ("gobierno", 0.13), ("debate", 0.10), ("partidos", 0.08)]
}
```

### Ejemplo de Salida
```json
{
  "0": "Problemas de salud en perros",
  "1": "Avances en tecnología e innovación",
  "2": "Debates y procesos políticos"
}
```

### Datos de Entrada
```json
{dict(lda_model.show_topics(num_words=25, formatted=False, num_topics=NUM_TOPICS))}
```
```

---

Este prompt asegura claridad en las instrucciones, incluye ejemplos útiles para contextualizar el proceso y garantiza que el modelo se enfoque en inferir títulos precisos y relevantes a partir de las palabras predominantes en cada tema.
"""


# TODO: Mejorar prompt
def topic_names_gpt(
    lda_model, NUM_TOPICS, prompt=None, key=None, contexto="", **kwargs
):
    """
    Kwargs gor  more paramters.
    """
    if key:
        if not prompt:

            prompt = f"""
You are a data scientist who uses Latent Dirichlet models.
Your task is to name the topics given by the output of an LDA model.

Context: {contexto}

You will receive a dictionary where the keys are Topic IDs and the values are lists of tuples 
containing words and their probabilities. Higher probability means higher importance of the word 
for each topic.

Rules:
1. Analyze all words belonging to the same topic
2. Create a descriptive title that captures the theme
3. Do not include the specific words from the list in the title
4. Ensure the response relates to the given words
5. Return exactly {NUM_TOPICS} topics
6. The response must be in Spanish

Return the response as a properly formatted JSON with this exact structure:
{{
    "0": "Description of topic 0",
    "1": "Description of topic 1",
    "2": "Description of topic 2"
}}

Input data:
{dict(lda_model.show_topics(num_words=25, formatted=False, num_topics=NUM_TOPICS))}"""

        response = get_completion(prompt=prompt, key=key)
        return response
    else:
        print("no opena AI key")


def get_completion(prompt, model="gpt-4o", key=""):
    """
    Generates text completions with improved error handling.

    Args:
        prompt (str): The input text prompt
        model (str): The GPT model to use
        key (str): OpenAI API key

    Returns:
        dict: Parsed JSON response or empty dict on error
    """
    if not key:
        logger.error("No OpenAI API key provided")
        return {}
        
    try:
        client = OpenAI(api_key=key)
        
        response = client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.2,
            max_tokens=1000  # Add token limit for cost control
        )

        content = response.choices[0].message.content
        
        # Clean JSON response
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = content.rstrip("`").strip()
        
        # Validate JSON structure
        content_data = json.loads(content)
        
        # Ensure all topics are present
        if isinstance(content_data, dict):
            return content_data
        else:
            logger.error(f"Invalid response format: expected dict, got {type(content_data)}")
            return {}
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Raw content: {content if 'content' in locals() else 'No content'}")
        return {}
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return {}
