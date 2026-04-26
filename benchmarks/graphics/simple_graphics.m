JSON_FILE = 'data.json';

fid = fopen(JSON_FILE, 'r');
if fid == -1
  error('Could not open file: %s', JSON_FILE);
end
raw = fread(fid, Inf, '*char')';
fclose(fid);

parsed = jsondecode(raw);

if iscell(parsed)
  parsed = parsed{1};
end


n = numel(parsed);

bubble_id              = zeros(n, 1);
extraction_symbols     = zeros(n, 1);
extraction_time_s      = zeros(n, 1);
translation_symbols    = zeros(n, 1);
translating_time_s     = zeros(n, 1);
typesetting_characters = zeros(n, 1);
typesetting_time_s     = zeros(n, 1);

for i = 1:n
  bubble_id(i)              = parsed(i).id;
  extraction_symbols(i)     = parsed(i).extraction_symbols;
  extraction_time_s(i)      = parsed(i).extraction_time_s;
  translation_symbols(i)    = parsed(i).translation_symbols;
  translating_time_s(i)     = parsed(i).translating_time_s;
  typesetting_characters(i) = parsed(i).typesetting_characters;
  typesetting_time_s(i)     = parsed(i).typesetting_time_s;
end

printf('Carregadas %d bolhas de %s\n', n, JSON_FILE);

mask_ext  = extraction_symbols     > 0;
mask_tra  = translation_symbols    > 0;
mask_typ  = typesetting_characters > 0;

p_ext = polyfit(extraction_symbols(mask_ext),     extraction_time_s(mask_ext),     1);
p_tra = polyfit(translation_symbols(mask_tra),    translating_time_s(mask_tra),    1);
p_typ = polyfit(typesetting_characters(mask_typ), typesetting_time_s(mask_typ),    1);

r2 = @(y_real, x, p) 1 - sum((y_real - polyval(p, x)).^2) / sum((y_real - mean(y_real)).^2);

fit_line = @(x, p) deal( ...
  linspace(min(x), max(x), 200), ...
  polyval(p, linspace(min(x), max(x), 200)) ...
);

% --- 1) Extração ---
f1 = figure('Visible', 'off');
scatter(extraction_symbols(mask_ext), extraction_time_s(mask_ext), 60, 'b', 'filled');
hold on;
[xf, yf] = fit_line(extraction_symbols(mask_ext), p_ext);
plot(xf, yf, 'b--', 'LineWidth', 1.5);
r2_ext = r2(extraction_time_s(mask_ext), extraction_symbols(mask_ext), p_ext);
xlabel('Quantidade de ideogramas/Simbolos');
ylabel('Tempo (s)');
title(sprintf('Extração  |  y = %.5f·x + %.5f  |  R² = %.3f', p_ext(1), p_ext(2), r2_ext));
legend('Bolhas', 'Tendência linear', 'Location', 'northwest');
xlim([0, max(extraction_symbols(mask_ext)) * 1.05]);
ylim([0, max(extraction_time_s(mask_ext)) * 1.1]);
grid on;
saveas(gcf, 'extracao.png');

% --- 2) Tradução ---
f2 = figure('Visible', 'off');
scatter(translation_symbols(mask_tra), translating_time_s(mask_tra), 60, 'r', 'filled');
hold on;
[xf, yf] = fit_line(translation_symbols(mask_tra), p_tra);
plot(xf, yf, 'r--', 'LineWidth', 1.5);
r2_tra = r2(translating_time_s(mask_tra), translation_symbols(mask_tra), p_tra);
xlabel('Quantidade de ideogramas/Simbolos');
ylabel('Tempo (s)');
title(sprintf('Tradução  |  y = %.5f·x + %.5f  |  R² = %.3f', p_tra(1), p_tra(2), r2_tra));
legend('Bolhas', 'Tendência linear', 'Location', 'northwest');
xlim([0, max(translation_symbols(mask_tra)) * 1.05]);
ylim([0, max(translating_time_s(mask_tra)) * 1.1]);
grid on;
saveas(gcf, 'traducao.png');

% --- 3) Typesetting ---
f3 = figure('Visible', 'off');
scatter(typesetting_characters(mask_typ), typesetting_time_s(mask_typ), 60, [0.1 0.7 0.1], 'filled');
hold on;
[xf, yf] = fit_line(typesetting_characters(mask_typ), p_typ);
plot(xf, yf, 'g--', 'LineWidth', 1.5);
r2_typ = r2(typesetting_time_s(mask_typ), typesetting_characters(mask_typ), p_typ);
xlabel('Quantidade de caracteres');
ylabel('Tempo (s)');
title(sprintf('Typesetting  |  y = %.5f·x + %.5f  |  R² = %.3f', p_typ(1), p_typ(2), r2_typ));
legend('Bolhas', 'Tendência linear', 'Location', 'northwest');
xlim([0, max(typesetting_characters(mask_typ)) * 1.05]);
ylim([0, max(typesetting_time_s(mask_typ)) * 1.1]);
grid on;
saveas(gcf, 'typesetting.png');

close(f1); close(f2); close(f3);

printf('\n=== EXTRAÇÃO (%d bolhas) ===\n', sum(mask_ext));
printf('  Intervalo de símbolos : %d – %d\n',   min(extraction_symbols(mask_ext)),  max(extraction_symbols(mask_ext)));
printf('  Intervalo de tempo    : %.4f – %.4f s\n', min(extraction_time_s(mask_ext)), max(extraction_time_s(mask_ext)));
printf('  Tempo médio           : %.4f s\n',    mean(extraction_time_s(mask_ext)));
printf('  Ajuste linear         : tempo = %.6f * símbolos + %.6f\n', p_ext(1), p_ext(2));
printf('  R²                    : %.4f\n', r2_ext);

printf('\n=== TRADUÇÃO (%d bolhas) ===\n', sum(mask_tra));
printf('  Intervalo de símbolos : %d – %d\n',   min(translation_symbols(mask_tra)),  max(translation_symbols(mask_tra)));
printf('  Intervalo de tempo    : %.4f – %.4f s\n', min(translating_time_s(mask_tra)), max(translating_time_s(mask_tra)));
printf('  Tempo médio           : %.4f s\n',    mean(translating_time_s(mask_tra)));
printf('  Ajuste linear         : tempo = %.6f * símbolos + %.6f\n', p_tra(1), p_tra(2));
printf('  R²                    : %.4f\n', r2_tra);

printf('\n=== TYPESETTING (%d bolhas) ===\n', sum(mask_typ));
printf('  Intervalo de caracteres : %d – %d\n', min(typesetting_characters(mask_typ)), max(typesetting_characters(mask_typ)));
printf('  Intervalo de tempo      : %.4f – %.4f s\n', min(typesetting_time_s(mask_typ)), max(typesetting_time_s(mask_typ)));
printf('  Tempo médio             : %.4f s\n',  mean(typesetting_time_s(mask_typ)));
printf('  Ajuste linear           : tempo = %.6f * caracteres + %.6f\n', p_typ(1), p_typ(2));
printf('  R²                      : %.4f\n', r2_typ);

printf('\n=== BOLHAS IGNORADAS (sem símbolos) ===\n');
skipped = bubble_id(~mask_ext);
if isempty(skipped)
  printf('  Nenhuma\n');
else
  printf('  IDs: ');
  printf('%d ', skipped);
  printf('\n');
end