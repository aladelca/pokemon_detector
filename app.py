import streamlit as st
import torch
from PIL import Image
import os
from pokemon_models import PokemonResNet, predict_image
from config import resnet_transform

# Configuración de la página
st.set_page_config(
    page_title="Pokémon Detector",
    page_icon="🎮",
    layout="centered"
)

# Título y descripción
st.title("Pokémon Detector")
st.markdown("""
Esta aplicación utiliza un modelo ResNet pre-entrenado para identificar Pokémon en imágenes.
Sube una imagen y el modelo te dirá qué Pokémon es y con qué confianza lo ha identificado.
""")

# Ruta al modelo pre-entrenado
MODEL_PATH = "models/pokemon_resnet_model.pth"

def get_idx_to_class_from_data_dir(data_dir):
    """Reconstruye el mapeo idx_to_class leyendo los nombres de las carpetas en el directorio de datos."""
    if not os.path.exists(data_dir):
        st.warning(f"No se encontró el directorio de datos: {data_dir}")
        return {}
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return {i: c for i, c in enumerate(classes)}

# Cambia este path si tu dataset está en otro lugar
DATA_DIR = "data"

def load_model():
    """Carga el modelo pre-entrenado"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el modelo en {MODEL_PATH}")
        return None
    
    try:
        # Cargar el modelo completo con weights_only=False
        loaded_data = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        
        # Verificar si el objeto cargado es el modelo completo
        if isinstance(loaded_data, PokemonResNet):
            model = loaded_data
            model.eval()
            idx_to_class = get_idx_to_class_from_data_dir(DATA_DIR)
            return model, idx_to_class, resnet_transform
        
        # Si no es el modelo completo, verificar si es un diccionario
        if isinstance(loaded_data, dict):
            # Si es un diccionario, extraer el estado del modelo y las clases
            model_state = loaded_data.get('model_state_dict', loaded_data)
            idx_to_class = loaded_data.get('idx_to_class', {})
            
            # Si no hay mapeo, reconstruirlo
            if not idx_to_class:
                idx_to_class = get_idx_to_class_from_data_dir(DATA_DIR)
            
            # Crear el modelo
            model = PokemonResNet(len(idx_to_class) if idx_to_class else 151)
            model.load_state_dict(model_state)
            model.eval()
            
            return model, idx_to_class, resnet_transform
        else:
            st.error("Formato de modelo no reconocido")
            return None
            
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocesa la imagen para el modelo"""
    try:
        # Convertir a RGB si no lo es
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Aplicar las transformaciones
        image_tensor = resnet_transform(image)
        return image_tensor.unsqueeze(0)  # Añadir dimensión de batch
    except Exception as e:
        st.error(f"Error al preprocesar la imagen: {str(e)}")
        return None

def main():
    # Cargar el modelo
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    model, idx_to_class, transform = model_data
    
    # Sección para subir imagen
    st.header("Sube una imagen de un Pokémon")
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Mostrar la imagen subida
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)
        
        # Botón para realizar la predicción
        if st.button("Identificar Pokémon"):
            with st.spinner("Analizando imagen..."):
                # Preprocesar la imagen
                image_tensor = preprocess_image(image)
                if image_tensor is None:
                    st.stop()
                
                # Realizar la predicción
                try:
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        predicted_class = outputs.argmax(1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    # Mostrar resultados
                    st.success(f"¡Predicción exitosa!")
                    st.markdown(f"### Resultados:")
                    
                    # Manejar el caso donde no tenemos mapeo de clases
                    if idx_to_class:
                        pokemon_name = idx_to_class.get(predicted_class, f"Pokémon #{predicted_class}")
                    else:
                        pokemon_name = f"Pokémon #{predicted_class}"
                    
                    st.markdown(f"**Pokémon identificado:** {pokemon_name}")
                    st.markdown(f"**Confianza:** {confidence:.2%}")
                    
                    # Mostrar barra de progreso para la confianza
                    st.progress(confidence)
                    
                except Exception as e:
                    st.error(f"Error al realizar la predicción: {str(e)}")

if __name__ == "__main__":
    main() 