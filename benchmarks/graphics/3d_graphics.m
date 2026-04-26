fid = fopen('data3d.json', 'r');
conteudo = fread(fid, '*char')';
fclose(fid);

% Decodificar JSON
dados = jsondecode(conteudo);

% Extrair as 3 variáveis de interesse
x = [dados.detection_rects_total];
y = [dados.area];
z = [dados.detection_time_s];

% Remover possíveis valores inválidos (se houver)
idx = isfinite(x) & isfinite(y) & isfinite(z);
x = x(idx);
y = y(idx);
z = z(idx);

% Preparar grid para interpolação
N = 50;  % resolução da grade
xi = linspace(min(x), max(x), N);
yi = linspace(min(y), max(y), N);
[XI, YI] = meshgrid(xi, yi);

% Interpolação (exemplo: método linear)
ZI = griddata(x, y, z, XI, YI, 'linear');

% Tratar NaNs na interpolação (substituir por média)
ZI(isnan(ZI)) = mean(z);

% --- Figura 1: Heatmap ---
figure(1)
imagesc(xi, yi, ZI)
axis xy
colorbar
colormap(jet)
xlabel('Quantidade de retângulos (detection\_rects\_total)')
ylabel('Área do retângulo (pixels)')
title('Heatmap: Tempo de detecção')
grid on
print(1, 'heatmap_bins.png', '-dpng', '-r600')

% --- Figura 2: Surface 3D ---
figure(2)
surf(XI, YI, ZI, 'EdgeColor', 'none')
colormap(jet)
colorbar
xlabel('Quantidade de retângulos')
ylabel('Área do retângulo')
zlabel('Tempo de detecção (s)')
title('Superfície 3D: detection\_time\_s vs area vs detection\_rects\_total')
view(45, 30)
grid on
print(2, 'grafico_3d_pontos.png', '-dpng', '-r600')