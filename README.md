# Proyecto de Procesamiento de Lenguaje Natural con Hugging Face

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este proyecto demuestra diferentes capacidades de procesamiento de lenguaje natural utilizando modelos de Hugging Face Transformers.

## Características

- 📝 Clasificación de texto (Zero-shot)
- 🤖 Generación de texto
- 😊 Análisis de sentimientos
- 📚 Resumen de textos
- 🌍 Traducción (Español a Inglés)

## Requisitos

```bash
pip install -r requirements.txt
```

Contenido del `requirements.txt`:
```txt
transformers==4.37.2
torch==2.1.2
sentencepiece==0.1.99
sacremoses==0.1.1
```

## Modelos Utilizados

- **Clasificación**: `facebook/bart-large-mnli`
- **Generación**: `distilgpt2`
- **Sentimientos**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Resumen**: `facebook/bart-large-cnn`
- **Traducción**: `Helsinki-NLP/opus-mt-es-en`

## Estructura del Código

```python
huggingfacecourse.py
├── inicializar_clasificador()      # Inicializa el clasificador de texto
├── clasificar_texto()              # Clasifica texto en categorías
├── inicializar_generador()         # Inicializa el generador de texto
├── generar_texto()                 # Genera texto a partir de un prompt
├── inicializar_analizador_sentimientos() # Inicializa analizador de sentimientos
├── analizar_sentimiento()          # Analiza el sentimiento de un texto
├── inicializar_resumidor()         # Inicializa el resumidor de texto
├── resumir_texto()                 # Genera resúmenes de textos largos
├── inicializar_traductor()         # Inicializa el traductor
└── traducir_texto()                # Traduce texto de español a inglés
```

## Uso

Para ejecutar el programa:

```bash
python huggingfacecourse.py
```

### Ejemplos de Uso

1. **Clasificación de Texto**:
   ```python
   texto = "Este curso trata sobre la biblioteca Transformers"
   etiquetas = ["educación", "política", "negocios"]
   ```

2. **Generación de Texto**:
   ```python
   texto_inicial = "En un futuro lejano,"
   # Genera una continuación del texto
   ```

3. **Análisis de Sentimientos**:
   ```python
   texto = "Me encanta este nuevo producto, es increíble!"
   # Analiza el sentimiento (positivo/negativo)
   ```

4. **Resumen de Texto**:
   ```python
   texto_largo = "La inteligencia artificial ha revolucionado..."
   # Genera un resumen conciso
   ```

5. **Traducción**:
   ```python
   texto = "La tecnología está cambiando nuestro mundo"
   # Traduce al inglés
   ```

## Configuración de GPU

El código detecta automáticamente si hay una GPU disponible y la utiliza para acelerar el procesamiento. Si no hay GPU, utilizará la CPU.

## Notas

- Los modelos se descargarán automáticamente en la primera ejecución
- Algunos modelos pueden ser grandes, asegúrate de tener suficiente espacio en disco
- La primera ejecución puede ser más lenta debido a la descarga de modelos

## Contribuir

Siéntete libre de abrir issues o enviar pull requests con mejoras.

## Licencia

MIT License
