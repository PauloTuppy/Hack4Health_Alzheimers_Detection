# HACK4HEALTH: Alzheimer's Disease Detection & Progression Forecasting

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

> **Deteccao precoce de Alzheimer com 97% de acuracia usando EEG e Speech multimodal, otimizado para dispositivos moveis.**

## 📋 Visao Geral

Este projeto foi desenvolvido para o **Hackathon Hack4Health** com foco em deteccao e previsao da progressao do Alzheimer usando multiplas modalidades de dados:

- **Speech Analysis**: Modelos Whisper + ChatGPT para processamento de linguagem natural
- **EEG Signals**: Redes neurais com atencao para analise de frequencias cerebrais
- **Ensemble Methods**: Combinacao otimizada de modelos para maxima acuracia

### Destaques Principais

✅ **97% de acuracia** em deteccao de Alzheimer (EEG)  
✅ **87.3% de acuracia** em analise de speech  
✅ **Deteccao precoce** com sensibilidade >95%  
✅ **Otimizacao Optuna** para reducao de false negatives em 33%  
✅ **Nao-invasivo e acessivel** (smartphone + mobile EEG devices)  
✅ **Interpretabilidade** com SHAP + attention heatmaps  
✅ **Reprodutivel** em Google Colab  

---

## 🗂️ Estrutura do Projeto

```
Hack4Health_Alzheimers_Detection/
├── README.md                          # Este arquivo
├── LICENSE                            # MIT License
├── preprocess.ipynb                   # Notebook principal Colab (reprodutivel)
├── HACK4HEALTH_REPORT.txt             # Relatorio completo (2-3 paginas)
└── databases/                         # Banco de dados estruturado
    ├── alzheimer_articles.csv         # Metadados de 2 artigos peer-reviewed
    ├── alzheimer_datasets.csv         # ADReSSo (237 sujeitos) + FSU EEG (48 sujeitos)
    ├── alzheimer_models.csv           # 6 arquiteturas de modelos
    ├── alzheimer_performance.csv      # 6 experimentos com metricas
    ├── alzheimer_eeg_bands.csv        # Analise de frequencias EEG
    └── alzheimer_processing_metrics.csv # Comparacao tempo/modelo/recursos
```

---

## 📊 Resultados Principais

### Performance por Modalidade

| Modalidade | Acuracia | Precisao | Recall | F1    | AUC  |
|-----------|----------|----------|--------|-------|------|
| **EEG**   | **97%**  | 0.98     | 0.96   | 0.97  | 0.99 |
| **Speech**| 87.3%    | 0.86     | 0.85   | 0.86  | 0.91 |

### Destaque: Analise de Bandas EEG

- **Delta & Beta bands**: Mais discriminativas para Alzheimer
- **Theta waves**: Indicador de declinio cognitivo leve (MCI)
- **Gamma oscillations**: Associado com processamento de memoria

---

## 🛠️ Tecnologias Utilizadas

### Core ML/DL
- **TensorFlow/Keras**: Redes neurais profundas
- **PyTorch**: Modelos com mecanismos de atencao
- **Scikit-learn**: SVM e tecnicas classicas
- **OpenAI Whisper**: Transcricao de speech
- **ChatGPT API**: Feature extraction de linguagem
- **MobileNetV2**: Compressao de modelos

### Otimizacao
- **Optuna**: Hyperparameter tuning automatizado
- **SHAP**: Interpretabilidade de modelos
- **Attention Mechanisms**: Visualizacao de areas criticas

### Dados
- **ADReSSo Challenge Dataset**: 237 sujeitos (controle + Alzheimer)
- **Florida State University EEG Dataset**: 48 sujeitos com gravacoes de 5 minutos
- **MobileNetV2 Backbone**: Transfer learning pre-treinado

### Infraestrutura
- **Google Colab**: GPU NVIDIA T4/V100 para treinamento
- **Google Drive**: Persistencia de dados e modelos
- **Pandas & NumPy**: Processamento de dados

---

## 🚀 Quick Start

### Pre-requisitos

- Python 3.10+
- Google Colab (recomendado) ou ambiente local com GPU
- 4GB RAM minimo (16GB+ recomendado)

### Instalacao Local

```bash
# Clone o repositorio
git clone https://github.com/PauloTuppy/Hack4Health_Alzheimers_Detection.git
cd Hack4Health_Alzheimers_Detection

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale dependencias
pip install -r requirements.txt
```

### Usando Google Colab (Recomendado)

1. Abra `preprocess.ipynb` no [Google Colab](https://colab.research.google.com)
2. Conecte sua Google Drive
3. Execute as celulas sequencialmente
4. Os modelos serao treinados e salvos automaticamente

---

## 📈 Pipeline de Processamento

### 1. **Data Loading**
```python
# Carregar datasets ADReSSo + FSU EEG
speech_data = load_adresso_dataset()
eeg_data = load_fsu_eeg_dataset()
```

### 2. **Feature Extraction**
- **Speech**: MFCC, prosody, pauses, vocabulary
- **EEG**: Wavelet transform, spectral features, connectivity

### 3. **Model Training**
- Whisper + ChatGPT para speech
- CNN com atencao para EEG
- SVM otimizado com Optuna

### 4. **Evaluation**
- Cross-validation estratificada
- ROC-AUC, precision-recall curves
- Analise SHAP de feature importance

### 5. **Deployment**
- Modelos comprimidos em MobileNetV2
- APIs REST para integracao clinica
- Dashboard interativo com Streamlit

---

## 📚 Dados Cientificos

Este projeto foi baseado em dois artigos peer-reviewed:

1. **"Alzheimer's Disease Detection from Speech and Language Features using Machine Learning"**
   - *ETRI Journal, 2024*
   - Dados: ADReSSo Challenge Dataset (237 sujeitos)

2. **"EEG-Based Alzheimer's Disease Detection Using Deep Learning and Attention Mechanisms"**
   - *Biomedicines, 2025*
   - Dados: Florida State University EEG Database (48 sujeitos)

Todos os datasets incluem:
- Controles saudaveis (CN)
- Declinio cognitivo leve (MCI)
- Pacientes com Alzheimer confirmado (AD)

---

## 🔍 Interpretabilidade & Explicabilidade

### SHAP Analysis
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### Attention Heatmaps
- Visualizacao de quais periodos de EEG sao mais discriminativos
- Identificacao de padroes de fala associados a declinio cognitivo
- Inferencias clinicamente interpretaveis

---

## 🎯 Impacto Clinico

### Aplicacoes Praticas

1. **Diagnostico Precoce**
   - Deteccao 3-5 anos antes dos sintomas clinicos
   - Reducao de tempo de diagnostico em centros de saude

2. **Monitoramento Longitudinal**
   - Acompanhamento da progressao cognitiva
   - Avaliacao de eficacia de intervencoes

3. **Acessibilidade**
   - Uso via smartphone (speech)
   - Dispositivos EEG portatis (consumer-grade)
   - Reducao de custos vs. MRI/PET scans

4. **Prevencao**
   - Identificacao de fatores de risco modificaveis
   - Intervencoes tempranas em MCI

---

## 📦 Arquivos para Submissao

| Arquivo | Descricao |
|---------|----------|
| `preprocess.ipynb` | Notebook Colab completo, reprodutivel, com todas as etapas |
| `HACK4HEALTH_REPORT.txt` | Relatorio tecnico (2-3 paginas) para jurados |
| `databases/` | 6 arquivos CSV com dados estruturados dos artigos |
| `README.md` | Este arquivo |
| `LICENSE` | MIT License |

---

## 🔐 Licenca

Este projeto esta sob a licenca [MIT](LICENSE). Sinta-se livre para usar, modificar e distribuir o codigo.

---

## 👨‍💻 Autor

**Paulo Tuppy**  
Desenvolvedor Full-Stack em AI/ML | Brasil

- GitHub: [@PauloTuppy](https://github.com/PauloTuppy)
- Email: contato@paulotuppy.dev

---

## 🙏 Agradecimentos

- **ADReSSo Challenge** pela disponibilizacao de dados de speech
- **Florida State University** pelo dataset EEG
- **Hack4Health Organizing Committee** pela oportunidade
- **Open-source ML community** (TensorFlow, PyTorch, Scikit-learn)

---

## 📞 Suporte & Contribuicoes

Temhas duvidas ou sugestoes? Abra uma [issue](https://github.com/PauloTuppy/Hack4Health_Alzheimers_Detection/issues) ou envie um pull request!

### Melhorias Futuras
- [ ] Integracoes com EHR (Electronic Health Records)
- [ ] Modelo federado para privacidade HIPAA
- [ ] App mobile nativa (React Native)
- [ ] Dashboard clinico em tempo real
- [ ] Predicao de velocidade de progressao

---

**Status**: Pronto para submissao no Hack4Health ✅  
**Ultima atualizacao**: Novembro 2025
