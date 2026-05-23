# FTM — Ferramenta de Tradução de Mangás

Pipeline modular de tradução de mangás que detecta automaticamente balões de fala, extrai o texto em japonês, traduz e insere o resultado de volta na imagem.

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) do curso de Tecnologia em Análise e Desenvolvimento de Sistemas (TADS) do Instituto Federal do Rio Grande do Sul (IFRS). **O texto completo do trabalho estará disponível [aqui](URL_DO_TCC) após a defesa**.

---

## Índice

- [Demo](#demo)
- [Funcionalidades](#funcionalidades)
- [Início Rápido](#início-rápido)
- [Instalação](#instalação)
  - [Via uv (recomendado para uso como comando global)](#via-uv-recomendado-para-uso-como-comando-global)
  - [Via pipx (recomendado para uso como comando global)](#via-pipx-recomendado-para-uso-como-comando-global)
  - [Via pip (recomendado)](#via-pip-recomendado)
  - [Via clone local](#via-clone-local)
  - [Modelo de tradução (Ollama)](#modelo-de-tradução-ollama)
- [Referência da CLI](#referência-da-cli)
  - [Obrigatório](#obrigatório)
  - [Opcional](#opcional)
- [Exemplos](#exemplos)
- [Arquivos de Saída](#arquivos-de-saída)
- [Etapas do Pipeline](#etapas-do-pipeline)
- [Formato do `bubbles.json`](#formato-do-bubblesjson)
- [Datasets](#Datasets)
- [Aviso Legal (Disclaimer)](#aviso-legal-disclaimer)

---

## Demo

https://github.com/user-attachments/assets/1bcc40bb-ee92-44f4-a60e-54415038b997


![Demo](https://youtu.be/73rHTp9IfYo)

*Manga usado na Demo: Hamlet©Taro Minamoto via Manga109-s Dataset*

---

## Funcionalidades

- **Detecção de Balões** — Detecção via YOLO com modelo fine-tuned no dataset Manga109-s
- **Extração de Texto** — MangaOCR para reconhecimento preciso de japonês
- **Tradução** — Modelos Ollama (`translategemma:4b` / `translategemma:12b`)
- **Diagramação** — Renderiza o texto traduzido de volta nos balões
- **Pipeline Modular** — Execute todas as etapas ou apenas as que precisar
- **Controle CPU/GPU** — Configuração granular por etapa
- **Monitoramento de Recursos** — Perfilamento opcional de CPU/RAM salvo em CSV

---

## Início Rápido

```bash
python -m ftm --image ./pagina_manga.jpg
```

Executa o pipeline completo (detecção → extração → tradução → diagramação) e salva a imagem traduzida no diretório informado.

---

## Instalação

> **Requisitos:** Python 3.9+, [Ollama](https://ollama.com/) rodando localmente e, opcionalmente, uma GPU compatível com CUDA.


### Via uv (recomendado para uso como comando global)

**Instalação padrão:**
```bash
uv tool install "git+ssh://git@github.com/Chipskein/ftm.git" --python 3.11
```

### Via pipx (recomendado para uso como comando global)

**Instalação padrão:**
```bash
pipx install "git+ssh://git@github.com/Chipskein/ftm.git" --python /usr/bin/python3.11 --verbose
```

**Com suporte a GPU NVIDIA (nvidia-ml-py):**
```bash
pipx install "ftm[nvidia] @ git+ssh://git@github.com/Chipskein/ftm.git" --python /usr/bin/python3.11 --verbose
```

### Via pip (recomendado)

**Instalação padrão:**
```bash
pip install git+https://github.com/Chipskein/ftm.git
```

**Com suporte a GPU NVIDIA (nvidia-ml-py):**
```bash
pip install "ftm[nvidia] @ git+https://github.com/Chipskein/ftm.git"
```

**Versão específica ou branch:**
```bash
# branch específica
pip install git+https://github.com/Chipskein/ftm.git@main

# tag/versão específica
pip install git+https://github.com/Chipskein/ftm.git@v0.0.1
```

### Via clone local

```bash
git clone https://github.com/Chipskein/ftm.git
cd ftm

# instalação padrão
pip install .

# com suporte NVIDIA
pip install ".[nvidia]"
```

### Modelo de tradução (Ollama)

Após instalar, baixe o modelo de tradução via Ollama:

```bash
ollama pull translategemma:4b

ollama pull translategemma:12b
```

---

## Referência da CLI

```
python -m ftm --image <caminho> [opções]
```

### Obrigatório

| Argumento | Descrição |
|---|---|
| `--image CAMINHO` | Caminho para a imagem de entrada |

### Opcional

| Argumento | Padrão | Descrição |
|---|---|---|
| `--output CAMINHO` | `tmp/<nome>_translated.jpg` | Caminho para a imagem traduzida de saída |
| `--tmp DIR` | `./tmp` | Diretório para arquivos temporários e resultados intermediários |
| `--debug` | `False` | Ativa logs detalhados e salva todos os recortes e imagens de debug |
| `--steps` | todos | Etapas a executar: `detection` `extraction` `translation` `typesetting` |
| `--bubbles-json CAMINHO` | `None` | Carrega dados de balões de um JSON, pulando a detecção |
| `--translate-lang` | `en` | Idioma de destino: `en` ou `pt` |
| `--translate-model` | `translategemma:4b` | Modelo Ollama: `translategemma:4b` ou `translategemma:12b` |
| `--ollama-host URL` | `http://localhost:11434` | Host personalizado da API do Ollama |
| `--ollama-model-temperature` | `0.0` | Temperatura do modelo (0.0 = determinístico) |
| `--use-cpu-all` | `False` | Força o uso de CPU em todas as etapas |
| `--use-cpu-step` | `[]` | Força CPU em etapas específicas: `detection` `extraction` |
| `--use-monitor` | `False` | Salva uso de CPU/RAM em CSV no diretório tmp (requer `psutil`) |
| `--huggingface_online_check` | `False` | Verifica disponibilidade dos modelos do Hugging Face na inicialização |

---

## Exemplos

> **Nota:** As imagens utilizadas nos testes e exemplos visuais deste projeto são provenientes do dataset [Manga109-s](http://www.manga109.org), licenciado para uso acadêmico. Por restrições de direitos autorais, essas imagens não estão disponíveis neste repositório — os resultados completos do pipeline podem ser consultados no texto acadêmico do TCC.

**Traduzir para português:**
```bash
python -m ftm --image pagina.jpg --translate-lang pt
```

**Rodar apenas tradução e diagramação (usando JSON salvo anteriormente):**
```bash
python -m ftm --image pagina.jpg --bubbles-json ./tmp/pagina_bubbles.json --steps translation typesetting
```

**Rodar somente na CPU (sem GPU):**
```bash
python -m ftm --image pagina.jpg --use-cpu-all
```

**Modo debug com monitoramento de recursos:**
```bash
python -m ftm --image pagina.jpg --debug --use-monitor
```

**Host Ollama personalizado (ex: servidor remoto):**
```bash
python -m ftm --image pagina.jpg --ollama-host http://192.168.1.100:11434
```

---

## Arquivos de Saída

Após uma execução bem-sucedida, os seguintes arquivos são salvos no diretório `--tmp`:

| Arquivo | Descrição |
|---|---|
| `<nome>_translated.jpg` | Imagem final traduzida |
| `<nome>_bubbles.json` | Metadados dos balões: coordenadas, texto OCR, traduções e tempos |
| `resources_<nome>_yolo.csv` | Log de uso de recursos (se `--use-monitor` estiver ativo) |

---

## Etapas do Pipeline

```
Imagem de Entrada
       │
       ▼
┌─────────────┐
│   Detecção  │  YOLO detecta regiões de balões → recortes salvos em tmp/
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Extração  │  MangaOCR lê o texto japonês de cada recorte
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Tradução  │  LLM Ollama traduz JP → idioma de destino
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Diagramação │  Texto traduzido renderizado de volta na imagem original
└─────────────┘
       │
       ▼
Imagem de Saída + bubbles.json
```

Cada etapa pode ser executada de forma independente usando `--steps` e `--bubbles-json`.

---

## Formato do `bubbles.json`

O arquivo JSON intermediário armazena os dados de cada balão detectado:

```json
[
  {
    "id": 0,
    "x": 120,
    "y": 340,
    "w": 200,
    "h": 150,
    "crop": "./tmp/pagina_crop_0.jpg",
    "jp_text": "こんにちは！",
    "translated_text": "Olá!",
    "area": 30000,
    "detection_method": "yolo",
    "detection_rects_total": 8,
    "detection_rects_kept": 5,
    "detection_time_s": 0.21,
    "grouping_time_s": 0.03,
    "extraction_symbols": 6,
    "extraction_time_s": 0.42,
    "translation_symbols": 6,
    "translating_time_s": 1.13,
    "typesetting_characters": 4,
    "typesetting_time_s": 0.08
  }
]
```

---

## Datasets

### Dataset — Manga109-s
 
O modelo de detecção de balões foi ajustado com o dataset [Manga109-s](http://www.manga109.org), uma versão do Manga109 licenciada para uso acadêmico e de pesquisa.
 
```bibtex
@inproceedings{fujimoto2018manga109,
  title     = {Manga109 dataset and creation of metadata},
  author    = {Fujimoto, Azuma and Ogawa, Toru and Yamamoto, Kazuyoshi and Matsui, Yusuke and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
  booktitle = {Proceedings of the 1st Workshop on coMics ANalysis, Processing and Understanding (MANPU), ICPR},
  year      = {2018}
}
```
 
### Dataset — COO: Comic Onomatopoeia Dataset
 
```bibtex
@inproceedings{baek2022COO,
  title     = {COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts},
  author    = {Baek, Jeonghun and Matsui, Yusuke and Aizawa, Kiyoharu},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```
 
### Dataset — Manga109Dialog
 
```bibtex
@inproceedings{li2024manga109dialog,
  title     = {Manga109Dialog: A Large-scale Dialogue Dataset for Comics Speaker Detection},
  author    = {Li, Yingxuan and Aizawa, Kiyoharu and Matsui, Yusuke},
  booktitle = {Proceedings of the IEEE International Conference on Multimedia and Expo},
  year      = {2024}
}
```

---

## Aviso Legal (Disclaimer)

> **LEIA COM ATENÇÃO ANTES DE UTILIZAR ESTA FERRAMENTA.**

### Finalidade

O FTM — Ferramenta de Tradução de Mangás foi desenvolvido **exclusivamente para fins educacionais, de pesquisa, estudo de processamento de linguagem natural e uso pessoal não comercial**. Qualquer uso fora desse escopo é de responsabilidade exclusiva do usuário.

### Proibição de Uso para Scanlations

**É expressamente proibido** utilizar este software para produzir, reproduzir, distribuir, publicar ou comercializar traduções não autorizadas de obras protegidas por direitos autorais (*scanlations*).

A reprodução e distribuição de obras sem autorização expressa dos detentores dos direitos autorais pode configurar violação às seguintes legislações, a depender do país do usuário:

- **Brasil:** Lei nº 9.610/1998 (Lei de Direitos Autorais)
- **União Europeia:** Diretiva 2019/790 sobre Direitos Autorais no Mercado Único Digital
- **Estados Unidos:** Digital Millennium Copyright Act (DMCA) e 17 U.S.C. § 501
- **Japão:** Lei de Direitos Autorais (著作権法, Lei nº 48 de 1970)
- Demais legislações nacionais e tratados internacionais aplicáveis (Convenção de Berna, TRIPS)

### Isenção de Responsabilidade dos Desenvolvedores

OS DESENVOLVEDORES, CONTRIBUIDORES E MANTENEDORES DESTE PROJETO:

1. **Não se responsabilizam**, em nenhuma hipótese, por qualquer uso ilegal, indevido ou prejudicial do software;
2. **Não endossam, incentivam ou apoiam** qualquer uso desta ferramenta que viole direitos autorais ou qualquer outra legislação aplicável;
3. **Não fornecem garantias** de qualquer natureza sobre o software, expresso ou implícito, incluindo, sem limitação, garantias de adequação a uma finalidade específica;
4. **Não são responsáveis** por quaisquer danos diretos, indiretos, incidentais, especiais ou consequenciais decorrentes do uso ou incapacidade de uso do software.

### Responsabilidade do Usuário

**O uso desta ferramenta é de inteira e exclusiva responsabilidade do usuário.** Ao utilizar o FTM, o usuário declara e garante que:

- Está ciente das leis de direitos autorais vigentes em seu país;
- Utilizará a ferramenta apenas para fins legais e autorizados;
- Possui os direitos necessários sobre o material que processará com o software;
- Isenta os desenvolvedores de qualquer responsabilidade civil ou criminal decorrente do uso indevido.

### Respeite os Criadores

Se você aprecia um mangá, apoie os autores e editoras adquirindo as obras oficiais ou acessando plataformas de leitura legais disponíveis em seu país.

> **Ao fazer download, clonar, instalar ou utilizar este software, você declara ter lido, compreendido e concordado integralmente com este Aviso Legal.**

---
