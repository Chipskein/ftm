% 1. Carregar os dados
% Pulamos 1 linha (header) e 1 coluna (nome do arquivo)
try
    % Ajustado para o seu novo nome de arquivo
    dados_brutos = csvread('comparacao-yolo-easy-paddle.csv', 1, 1);
catch
    error('Certifique-se de que o arquivo comparacao-yolo-easy-paddle.csv esta na mesma pasta!');
end

% 2. Limpar colunas extras e garantir apenas numeros
% Agora sao 9 colunas (3 colunas para cada um dos 3 modelos)
if size(dados_brutos, 2) > 9
    dados_brutos = dados_brutos(:, 1:9);
end

% 3. Filtrar apenas linhas preenchidas
somas_linhas = sum(abs(dados_brutos), 2);
dados = dados_brutos(somas_linhas > 0, :);

% 4. Somar os Totais (3 modelos agora)
% Modelo EASY
tpE = sum(dados(:,1)); fpE = sum(dados(:,2)); fnE = sum(dados(:,3));
% Modelo PADDLE
tpP = sum(dados(:,4)); fpP = sum(dados(:,5)); fnP = sum(dados(:,6));
% Modelo YOLO 1024
tpY = sum(dados(:,7)); fpY = sum(dados(:,8)); fnY = sum(dados(:,9));

% 5. Funções de Cálculo (Mantendo a lógica de evitar divisão por zero)
% EASY
pE = tpE / (tpE + fpE + (tpE+fpE==0));
rE = tpE / (tpE + fnE + (tpE+fnE==0));
f1_E = 2 * (pE * rE) / (pE + rE + (pE+rE==0));

% PADDLE
pP = tpP / (tpP + fpP + (tpP+fpP==0));
rP = tpP / (tpP + fnP + (tpP+fnP==0));
f1_P = 2 * (pP * rP) / (pP + rP + (pP+rP==0));

% YOLO
pY = tpY / (tpY + fpY + (tpY+fpY==0));
rY = tpY / (tpY + fnY + (tpY+fnY==0));
f1_Y = 2 * (pY * rY) / (pY + rY + (pY+rY==0));

% 6. Organizar para o gráfico (Matriz 3x3: Métricas nas linhas, Modelos nas colunas)
metricas = [pE, pP, pY; rE, rP, rY; f1_E, f1_P, f1_Y];

% 7. Gerar o Gráfico
figure;
h = bar(metricas);
set(gca, 'XTickLabel', {'Precisao', 'Recall', 'F1-Score'});
ylabel('Indice (0.0 a 1.0)');
title(['Analise de Desempenho (Amostra n = ', num2str(size(dados, 1)), ' imagens)']);
legend('EASY', 'PADDLE', 'YOLO 1024', 'location', 'northeastoutside');
grid on;
ylim([0 1.2]);

saveas(gcf, 'comparacao-metricas-final.png');

% 8. Exibir resultados no Terminal
fprintf('\n================ RESULTADOS FINAIS ================\n');
fprintf('Imagens processadas: %d\n', size(dados, 1));
fprintf('---------------------------------------------------\n');
fprintf('MODELO EASY   | P: %.4f | R: %.4f | F1: %.4f\n', pE, rE, f1_E);
fprintf('MODELO PADDLE | P: %.4f | R: %.4f | F1: %.4f\n', pP, rP, f1_P);
fprintf('MODELO YOLO   | P: %.4f | R: %.4f | F1: %.4f\n', pY, rY, f1_Y);
fprintf('===================================================\n');