"""
Teste simples da função NLIEE Score
"""
import cv2
import numpy as np
from nliee import nliee_score

def test_nliee_images(low_light_path, enhanced_path):
    """
    Testa a função NLIEE com imagens reais
    
    Args:
        low_light_path: Caminho para imagem de baixa luminosidade
        enhanced_path: Caminho para imagem realçada
    """
    
    # Carregar imagens
    print("Carregando imagens...")
    low_img = cv2.imread(low_light_path)
    enhanced_img = cv2.imread(enhanced_path)
    
    if low_img is None:
        print(f"Erro: Não foi possível carregar {low_light_path}")
        return
    
    if enhanced_img is None:
        print(f"Erro: Não foi possível carregar {enhanced_path}")
        return
    
    # Converter BGR para RGB
    low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar se necessário (para mesma resolução)
    if low_img.shape != enhanced_img.shape:
        print("Redimensionando imagens para mesmo tamanho...")
        h, w = min(low_img.shape[0], enhanced_img.shape[0]), min(low_img.shape[1], enhanced_img.shape[1])
        low_img = cv2.resize(low_img, (w, h))
        enhanced_img = cv2.resize(enhanced_img, (w, h))
    
    # Normalizar para [0, 1]
    low_img = low_img.astype(np.float32) / 255.0
    enhanced_img = enhanced_img.astype(np.float32) / 255.0
    
    # Calcular score NLIEE
    print("Calculando NLIEE Score...")
    score = nliee_score(low_img, enhanced_img)
    
    # Mostrar resultados
    print(f"\n=== RESULTADOS ===")
    print(f"Imagem baixa luminosidade: {low_light_path}")
    print(f"Imagem realçada: {enhanced_path}")
    print(f"NLIEE Score: {score:.2f}/100")
    
    # Interpretação do score
    if score >= 80:
        print("✅ Excelente qualidade de realce")
    elif score >= 60:
        print("✅ Boa qualidade de realce")
    elif score >= 40:
        print("⚠️  Qualidade moderada")
    else:
        print("❌ Qualidade baixa - necessário melhorar")
    
    return score

if __name__ == "__main__":
    # Exemplo de uso - substitua pelos seus caminhos
    low_light_path = "001_lowlight.png"      # Coloque o caminho da sua imagem escura
    enhanced_path = "001_Enhance_2.png"     # Coloque o caminho da sua imagem realçada
    
    # Executar teste
    test_nliee_images(low_light_path, enhanced_path)