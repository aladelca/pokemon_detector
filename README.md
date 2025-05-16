# Pokémon Detector

Este proyecto implementa un sistema de clasificación de imágenes de Pokémon utilizando dos arquitecturas diferentes de redes neuronales: una CNN personalizada y ResNet50 con transfer learning. Además, incluye una interfaz web interactiva para realizar predicciones en tiempo real.

## Características

- **Dos modelos de clasificación**:
  - CNN personalizada con dos capas convolucionales
  - ResNet50 con transfer learning
- **Interfaz web interactiva** usando Streamlit
- **Preprocesamiento de imágenes** optimizado para cada modelo
- **Sistema de entrenamiento** con:
  - Early stopping
  - Learning rate scheduling
  - Validación en tiempo real
  - Barra de progreso para seguimiento del entrenamiento

## Estructura del Proyecto

```
pokemon_detector/
├── models/                      # Directorio para modelos guardados
│   ├── pokemon_cnn_model.pth    # Modelo CNN entrenado
│   └── pokemon_resnet_model.pth # Modelo ResNet entrenado
├── data/                        # Directorio para datos de entrenamiento
│   └── [clases de pokémon]/     # Subdirectorios por clase
├── pokemon_models.py            # Implementación de los modelos
├── config.py                    # Configuración y hiperparámetros
├── app.py                       # Interfaz web Streamlit
└── requirements.txt             # Dependencias del proyecto
```

## Requisitos

- Python 3.8+
- PyTorch 2.0.0+
- Streamlit 1.30.0+
- Otras dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd pokemon_detector
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento de Modelos

Para entrenar cualquiera de los dos modelos:

```bash
# Entrenar CNN
python pokemon_models.py --mode train_cnn

# Entrenar ResNet
python pokemon_models.py --mode train_resnet
```

### Predicciones desde Línea de Comandos

```bash
# Predicción con CNN
python pokemon_models.py --mode predict_cnn --model_path models/pokemon_cnn_model.pth --image_path ruta/a/imagen.jpg

# Predicción con ResNet
python pokemon_models.py --mode predict_resnet --model_path models/pokemon_resnet_model.pth --image_path ruta/a/imagen.jpg
```

### Interfaz Web

Para iniciar la interfaz web:

```bash
streamlit run app.py
```

La interfaz web permite:
1. Subir imágenes de Pokémon
2. Ver la imagen subida
3. Realizar predicciones
4. Ver resultados con nivel de confianza

## Configuración

Los hiperparámetros y configuraciones se encuentran en `config.py`:

- Tamaño de batch
- Tasa de aprendizaje
- Número de épocas
- Tamaños de imagen
- Parámetros de normalización
- Transformaciones de imagen

## Modelos

### CNN Personalizada
- Dos bloques convolucionales
- Dropout para regularización
- Capas fully connected para clasificación

### ResNet50
- Transfer learning desde modelo pre-entrenado
- Capas convolucionales congeladas
- Clasificador personalizado

## Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.