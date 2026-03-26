NOTES:

Espera imagens de alta resolução utilizar upscaler.py se a imagem tiver ruim

PanelDetector[OK] -> 
    Detecta os painéis e retorna as coordenadas (x,y,w,h) de cada painel

EngineOCR(PanelDetector)[NAO OK] -> 
    Detecta textos e gera BubbleZones(Objeto que representa o texto com informações sobre metadados)
    e faz o corte das imagens


Regras de Classificação e Hierarquia

    Regra Global: Se um quadrado toca 3 ou mais painéis, ele é marcado como Global.
    Não agrupa com os dos paineis

    Regra do Vácuo (Atualizada): Se o quadrado toca 1 único painel e o restante da sua área está no vácuo (fora de qualquer painel) → Global.

    Regra de Precedência: A classificação de "Global" anula qualquer tentativa de agrupamento local.

    Regra de Posse Unitária: Se o quadrado toca apenas 1 painel, ele pertence exclusivamente a ele.

    Regra do Painel Dominante: Se tocar 2 painéis, ele pertence ao que tiver a maior área de intersecção.

    Margem de Segurança: 
        Aplique um Padding interno (ex: -2px) antes de checar colisões para evitar "toques fantasmas" nas bordas.



Regras de Agrupamento (Clusters)

    Filtro de Origem: 
        Só podem ser agrupados quadrados que pertençam ao mesmo Painel Pai.

    Critério de Proximidade: 
        A distância Borda a Borda deve ser menor que o limite definido (5% do painel).

    Alinhamento Horizontal: Se o texto for normal (W>H), agrupe apenas se houver sobreposição no eixo Y.

    Alinhamento Vertical: Se o texto for vertical (H>W), agrupe apenas se houver sobreposição no eixo X.

    Proteção de Coluna: 
        Para textos verticais, a tolerância de distância horizontal deve ser menor que a vertical.
















Talvez desnecessário -> BubbleRefiner[OK]

Extractor[OK] -> 
    Extrai os textos das imagens e guarda os resultados no BubbleZone

EngineTranslator[OK]
    Realiza as traduções e altera o objeto BubbleZone com as traduções

TypeSetter[OK]
    Sobreescreve o texto na imagem original com a tradução, utilizando um blur no fundo