from transformers import pipeline
import torch

def inicializar_clasificador():
    # Especificamos el modelo explícitamente
    nombre_modelo = "facebook/bart-large-mnli"
    
    # Verificar si hay GPU disponible
    dispositivo = 0 if torch.cuda.is_available() else -1
    print(f"Dispositivo configurado para usar: {'gpu' if dispositivo == 0 else 'cpu'}")
    
    # Inicializar el clasificador
    clasificador = pipeline(
        "zero-shot-classification",
        model=nombre_modelo,
        device=dispositivo
    )
    return clasificador

def clasificar_texto(clasificador, texto, etiquetas):
    # Realizar la clasificación y obtener resultados
    resultado = clasificador(texto, candidate_labels=etiquetas)
    
    # Mostrar resultados de forma legible
    print(f"\nTexto analizado: {texto}")
    print("\nResultados de la clasificación:")
    for etiqueta, puntuacion in zip(resultado['labels'], resultado['scores']):
        print(f"{etiqueta}: {puntuacion:.2%}")
    
    return resultado

def inicializar_generador():
    # Especificamos el modelo explícitamente
    nombre_modelo = "distilgpt2"
    
    # Verificar si hay GPU disponible
    dispositivo = 0 if torch.cuda.is_available() else -1
    print(f"Dispositivo configurado para usar: {'gpu' if dispositivo == 0 else 'cpu'}")
    
    # Inicializar el generador de texto
    generador = pipeline(
        "text-generation",
        model=nombre_modelo,
        device=dispositivo
    )
    return generador

def generar_texto(generador, texto_inicial, max_length=100):
    # Realizar la generación de texto
    resultado = generador(
        texto_inicial,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    
    # Mostrar resultado
    print(f"\nTexto inicial: {texto_inicial}")
    print("\nTexto generado:")
    print(resultado[0]['generated_text'])
    
    return resultado

def inicializar_analizador_sentimientos():
    nombre_modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
    analizador = pipeline("sentiment-analysis", model=nombre_modelo)
    return analizador

def analizar_sentimiento(analizador, texto):
    resultado = analizador(texto)
    print(f"\nAnálisis de sentimiento para: {texto}")
    print(f"Sentimiento: {resultado[0]['label']}")
    print(f"Confianza: {resultado[0]['score']:.2%}")
    return resultado

def inicializar_resumidor():
    nombre_modelo = "facebook/bart-large-cnn"
    resumidor = pipeline("summarization", model=nombre_modelo)
    return resumidor

def resumir_texto(resumidor, texto, max_length=130, min_length=30):
    resultado = resumidor(texto, max_length=max_length, min_length=min_length)
    print(f"\nTexto original: {texto[:100]}...")
    print(f"Resumen: {resultado[0]['summary_text']}")
    return resultado

def inicializar_traductor():
    nombre_modelo = "Helsinki-NLP/opus-mt-es-en"
    traductor = pipeline("translation", model=nombre_modelo)
    return traductor

def traducir_texto(traductor, texto):
    resultado = traductor(texto)
    print(f"\nTexto original: {texto}")
    print(f"Traducción: {resultado[0]['translation_text']}")
    return resultado

def main():
    # Inicializar el clasificador
    clasificador = inicializar_clasificador()
    
    # Ejemplos de textos y etiquetas
    textos = [
        "Este curso trata sobre la biblioteca Transformers de Hugging Face",
        "La bolsa de valores tuvo una caída significativa hoy",
        "Los científicos descubren una nueva especie en el Amazonas"
    ]
    
    conjuntos_etiquetas = [
        ["educación", "política", "negocios"],
        ["finanzas", "tecnología", "deportes"],
        ["ciencia", "medio ambiente", "entretenimiento"]
    ]
    
    # Procesar cada texto con sus etiquetas correspondientes
    for texto, etiquetas in zip(textos, conjuntos_etiquetas):
        clasificar_texto(clasificador, texto, etiquetas)
        print("-" * 50)

    # Inicializar el generador
    generador = inicializar_generador()
    
    # Ejemplos de textos iniciales
    textos_iniciales = [
        "En un futuro lejano,",
        "La inteligencia artificial",
        "El descubrimiento más importante"
    ]
    
    # Generar texto para cada ejemplo
    for texto in textos_iniciales:
        generar_texto(generador, texto)
        print("-" * 50)

    print("\n=== ANÁLISIS DE SENTIMIENTOS ===")
    analizador = inicializar_analizador_sentimientos()
    textos_sentimientos = [
        "Me encanta este nuevo producto, es increíble!",
        "Este servicio es terrible, no lo recomiendo.",
        "El día está agradable, perfecto para pasear."
    ]
    for texto in textos_sentimientos:
        analizar_sentimiento(analizador, texto)
        print("-" * 50)

    print("\n=== RESÚMENES DE TEXTO ===")
    resumidor = inicializar_resumidor()
    texto_largo = """
    La inteligencia artificial ha revolucionado múltiples campos en los últimos años.
    Desde la medicina hasta la educación, pasando por la industria y el entretenimiento,
    los sistemas basados en IA están transformando la manera en que vivimos y trabajamos.
    Los avances en aprendizaje profundo han permitido crear sistemas más sofisticados
    que pueden realizar tareas cada vez más complejas con mayor precisión.
    """
    resumir_texto(resumidor, texto_largo)
    print("-" * 50)

    print("\n=== TRADUCCIÓN ===")
    traductor = inicializar_traductor()
    textos_traducir = [
        "La tecnología está cambiando nuestro mundo",
        "El futuro es emocionante",
        "La programación es divertida"
    ]
    for texto in textos_traducir:
        traducir_texto(traductor, texto)
        print("-" * 50)

if __name__ == "__main__":
    main()