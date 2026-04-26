% 1. Carregar os dados
% Pulamos 1 linha (header) e 1 coluna (nome do arquivo)
try
    dados_brutos = csvread('comparacao-yolo-1024-320-matrix-confusao-manual.csv', 1, 1);
catch
    error('Certifique-se de que o arquivo comparacao-yolo-1024-320-matrix-confusao-manual.csv esta na mesma pasta!');
end

% 2. Limpar a ultima coluna (Observacoes) e garantir apenas numeros
% Suas colunas numericas sao as 6 primeiras (3 para o 320px, 3 para o 1024px)
if size(dados_brutos, 2) > 6
    dados_brutos = dados_brutos(:, 1:6);
end

% 3. Filtrar apenas linhas que voce ja preencheu (amostra de 123)
somas_linhas = sum(abs(dados_brutos), 2);
dados = dados_brutos(somas_linhas > 0, :);

% 4. Somar os Totais
tp3 = sum(dados(:,1)); fp3 = sum(dados(:,2)); fn3 = sum(dados(:,3));
tp1 = sum(dados(:,4)); fp1 = sum(dados(:,5)); fn1 = sum(dados(:,6));

% 5. Funções de Cálculo (ajustadas para evitar divisao por zero)
p3 = tp3 / (tp3 + fp3 + (tp3+fp3==0));
r3 = tp3 / (tp3 + fn3 + (tp3+fn3==0));
f1_3 = 2 * (p3 * r3) / (p3 + r3 + (p3+r3==0));

p1 = tp1 / (tp1 + fp1 + (tp1+fp1==0));
r1 = tp1 / (tp1 + fn1 + (tp1+fn1==0));
f1_1 = 2 * (p1 * r1) / (p1 + r1 + (p1+r1==0));

% 6. Organizar para o gráfico
metricas = [p3, p1; r3, r1; f1_3, f1_1];

% 7. Gerar o Gráfico
figure;
h = bar(metricas);
set(gca, 'XTickLabel', {'Precisao', 'Recall', 'F1-Score'});
ylabel('Indice (0.0 a 1.0)');
% CORREÇÃO AQUI: num2str em vez de num2mstr
title(['Analise de Desempenho (Amostra n = ', num2str(size(dados, 1)), ' imagens)']);
legend('YOLO 320px', 'YOLO 1024px', 'location', 'northeastoutside');
grid on;
ylim([0 1.2]); % Espaço para a legenda
saveas(gcf, 'comparacao-yolo-1024-320-metricas.png');

% 8. Exibir resultados no Terminal para copiar para o texto do TCC
fprintf('\n================ RESULTADOS FINAIS ================\n');
fprintf('Imagens processadas: %d\n', size(dados, 1));
fprintf('---------------------------------------------------\n');
fprintf('MODELO 320px  | P: %.4f | R: %.4f | F1: %.4f\n', p3, r3, f1_3);
fprintf('MODELO 1024px | P: %.4f | R: %.4f | F1: %.4f\n', p1, r1, f1_1);
fprintf('===================================================\n');