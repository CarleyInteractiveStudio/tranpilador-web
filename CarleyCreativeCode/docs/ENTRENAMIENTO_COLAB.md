# Guía de Entrenamiento en Google Colab para Carley Creative Code

Para entrenar tu modelo con las 500,000 líneas de código, sigue estos pasos:

## 1. Preparación del Entorno
En una celda de Colab, instala las librerías necesarias:
```bash
!pip install transformers tokenizers torch accelerate
```

## 2. Cargar los Archivos
Sube la carpeta `CarleyCreativeCode` a tu Google Drive o directamente a la sesión de Colab. Asegúrate de colocar todos tus archivos de código `.js` o `.cc` en la carpeta `CarleyCreativeCode/data/`.

## 3. Entrenamiento del Tokenizer
Ejecuta el script de tokenización para que el modelo aprenda tu lenguaje específico:
```python
%env PYTHONPATH=.
!python3 CarleyCreativeCode/scripts/train_tokenizer.py
```

## 4. Entrenamiento del Modelo (Ajuste de Parámetros)
Para entrenar el modelo de **125M de parámetros** (aprox. 500MB de RAM), abre `CarleyCreativeCode/scripts/train_model.py` y asegúrate de que la configuración sea:

```python
vocab_size = 32000
n_embd = 768
n_layer = 12
n_head = 12
block_size = 512 # O 1024 si tienes mucha VRAM
batch_size = 8
```

Luego inicia el entrenamiento:
```python
!python3 CarleyCreativeCode/scripts/train_model.py
```

## 5. Recomendaciones
- **GPU:** Usa una GPU T4 (gratuita) o A100/L4 (pago) para acelerar el proceso.
- **Checkpointing:** El script guardará el modelo final como `carley_model.pt`. Puedes modificarlo para guardar cada ciertas épocas.
- **Datos:** Cuanto más variados sean tus ejemplos en `data/`, mejor corregirá y generará código el modelo.
