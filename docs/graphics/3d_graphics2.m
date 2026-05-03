% Limpeza de ambiente
clear; clc; close all;

% 1. Carregar Dados do JSON
fid = fopen('data3d.json', 'r');
conteudo = fread(fid, '*char')';
fclose(fid);
dados = jsondecode(conteudo);

% 2. Extrair variáveis focadas em AGRUPAMENTO
% X = Total de detecções brutas (antes)
% Y = Total de detecções mantidas (depois)
% Z = Tempo gasto apenas no agrupamento (grouping_time_s)
x = [dados.detection_rects_total]; 
y = [dados.detection_rects_kept];  
z = [dados.grouping_time_s];

% 3. Configuração do Gráfico de Agrupamento
figure('Color', [1 1 1]);

% Gráfico Principal: Eficiência do Filtro
% O tamanho das bolhas representa o tempo (Z), a cor a redução
scatter(x, y, 50, z, 'filled');
colorbar;
colormap('viridis');
ylabel('Retângulos Após Agrupamento (Y)');
xlabel('Retângulos Antes do Agrupamento (X)');
title('Análise de Eficiência: Agrupamento de Bounding Boxes');
grid on;

% Adicionar linha de identidade (X=Y) para referência
% Quanto mais longe desta linha para baixo, mais eficiente é o agrupamento
hold on;
max_val = max([max(x), max(y)]);
plot([0, max_val], [0, max_val], 'r--', 'LineWidth', 1.5);
legend('Deteções Processadas', 'Linha de Referência (Sem Agrupamento)', 'Location', 'northwest');

% 4. Estatísticas de Compressão para o Relatório
taxa_compressao = mean(x ./ y);
fprintf('--- Sumário de Agrupamento ---\n');
fprintf('Taxa Média de Compressão: %.2f x\n', taxa_compressao);
fprintf('Tempo Médio de Agrupamento: %.4f segundos\n', mean(z));

saveas(gcf, 'grafico_agrupamento.png');