# Carley Creative Code (CCC)

**Carley Creative Code** es un modelo de lenguaje de inteligencia artificial diseñado específicamente para el motor de videojuegos **Creative Engine**.

Este modelo está optimizado para:
- **Generación de código:** Crear scripts de movimiento, lógica de juego y sistemas complejos a partir de descripciones naturales.
- **Corrección de errores:** Detectar y arreglar fallos en el código de Creative Engine.
- **Eficiencia extrema:** Diseñado para consumir entre 1 y 3 GB de RAM, ideal para integración local en Windows.
- **Sintaxis en Español:** Soporte nativo para las palabras clave únicas del motor (`si`, `sino`, `publico`, `variable`, etc.).

## Estructura del Proyecto

- `CarleyCreativeCode/src/`: Código fuente de la arquitectura del modelo y utilidades.
- `CarleyCreativeCode/scripts/`: Scripts para entrenamiento, tokenización e inferencia.
- `CarleyCreativeCode/docs/`: Documentación detallada del sistema.
- `CarleyCreativeCode/data/`: Directorio para colocar los datasets de entrenamiento (500k+ líneas).
- `training/`: Acceso rápido al flujo de entrenamiento.

## Características Técnicas

- **Arquitectura:** Transformer Decoder-only (estilo GPT).
- **Parámetros:** ~125 Millones.
- **Técnica de Aprendizaje:** Fill-In-The-Middle (FIM) para edición y corrección de código.
- **Framework:** PyTorch.
- **Tokenización:** BPE (Byte-Level Byte Pair Encoding) personalizado.

## Autoría
Desarrollado para el ecosistema de **Creative Engine**.
