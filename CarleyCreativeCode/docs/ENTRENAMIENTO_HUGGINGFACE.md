# Entrenamiento y Despliegue en Hugging Face (HF)

Para entrenar **Carley Creative Code** en Hugging Face, tienes dos opciones principales:

## Opción 1: Entrenar en un Space con GPU
Hugging Face Spaces permite alquilar GPUs potentes para entrenar:
1. Crea un **New Space** en tu cuenta de Hugging Face.
2. Sube todos los archivos de esta carpeta (`src/`, `scripts/`, `data/`, etc.).
3. Instala los requisitos (`pip install transformers accelerate tokenizers torch`).
4. Ejecuta el entrenamiento: `python scripts/train_model.py`.

## Opción 2: Subir el Modelo a Hugging Face Hub
Una vez que entrenes el modelo (ya sea en Colab o localmente), puedes subirlo al Hub para que sea accesible desde cualquier lugar:

### 1. Iniciar Sesión en HF
```bash
huggingface-cli login
```

### 2. Usar el Script de Subida
He creado un script especial (`scripts/push_to_hub.py`) para que subas tu modelo con un solo comando:
```bash
python scripts/push_to_hub.py --repo_id "tu_usuario/carley-creative-code"
```

### 3. Uso Compartido
Una vez subido, cualquier persona con permiso podrá cargar tu modelo usando:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("tu_usuario/carley-creative-code", trust_remote_code=True)
```

## Beneficios de Hugging Face
- **Control de Versiones:** Cada vez que mejores el modelo con más código, puedes subir una nueva versión.
- **Inferencia Gratuita:** Puedes crear una API básica en Hugging Face para que tu motor de videojuegos llame al modelo a través de internet sin consumir RAM local.
- **Créditos:** Tu modelo tendrá su propia página oficial con tu nombre y descripción.
