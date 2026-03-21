# Uso de Carley Creative Code (CCC) en Windows

Para integrar el modelo en tu motor **Creative Engine** o usarlo localmente en Windows, sigue estos pasos:

## 1. Instalación de Requisitos en Windows
Asegúrate de tener Python 3.10+ y PyTorch (CPU o CUDA) instalado:
```bash
pip install torch transformers tokenizers
```

## 2. Ejecución Local de Inferencia
Puedes usar el script proporcionado en la carpeta `scripts` para generar o corregir código:

### Generar Código
Para pedirle al modelo que cree un script de movimiento:
```bash
python scripts/inference_test.py --prompt "alActualizar(delta) { // codigo de salto "
```

### Corregir Código (FIM)
Para arreglar un trozo de código mal escrito en medio de un script:
```bash
python scripts/inference_test.py --prefix "si (teclaPresionada('Space'))" --suffix "fuerzaSalto = 15; }"
```

## 3. Integración en el Motor (Creative Engine)
Si tu motor está en C++ o C#, puedes usar librerías como:
- **LibTorch:** La versión en C++ de PyTorch.
- **ONNX Runtime:** Puedes convertir el modelo `.pt` a `.onnx` para una ejecución mucho más rápida y ligera en Windows.

### Conversión a ONNX (Opcional)
Para una eficiencia máxima (menos de 1GB de RAM), se recomienda convertir el modelo:
```python
import torch
# ... cargar modelo ...
dummy_input = torch.zeros(1, 1024, dtype=torch.long)
torch.onnx.export(model, dummy_input, "carley_model.onnx")
```
El motor puede cargar este archivo `.onnx` consumiendo muy pocos recursos.
