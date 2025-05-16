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

def load_model():
    """Carga el modelo pre-entrenado"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el modelo en {MODEL_PATH}")
        return None
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='mps')
        model = PokemonResNet(len(checkpoint['idx_to_class']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint['idx_to_class'], checkpoint['transform']
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
                    st.markdown(f"**Pokémon identificado:** {idx_to_class[predicted_class]}")
                    st.markdown(f"**Confianza:** {confidence:.2%}")
                    
                    # Mostrar barra de progreso para la confianza
                    st.progress(confidence)
                    
                except Exception as e:
                    st.error(f"Error al realizar la predicción: {str(e)}")

if __name__ == "__main__":
    main() 