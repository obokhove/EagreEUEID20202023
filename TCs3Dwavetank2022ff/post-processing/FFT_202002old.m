%_________________________________________________________
clear
close all

%_________________________________________________________
probe='1';
scheme='SV-GLL'; % MMP-VP or SV-GLL

p=str2num(probe);

% Absolute adata_path (change to your settings), e.g.:
  adata_path='/Users/amtob/werkvuurdraak2025/Wavetank2022-2025-main/';
% Main directory one:
file_exp = append('202002/WAVE_',probe,'.dat');
% Absolute path: file_exp = fullfile(adata_path, '202002', append('WAVE_', probe, '.dat'));
Data_exp = load(file_exp);

if scheme=='MMP-VP'
  % Old: Data_num = load('data_VP/MMP/TC4_2D(1ele,phiGLL)_5Jan/probes.csv');
  % Main directory path:
  Data_num = load('data/MMP/TC4_test/probes.csv');
  % New absolute path:
  % Data_num = fullfile(adata_path, 'data', 'MMP', 'TC4_test', 'probes.csv');
else
  % Old: Data_num = load('data_Re/SV/TC4_SV_GLL/probes.csv');
  % Main directory path:
  Data_num = load('data/SV/TC4_test/probes.csv');
  % New absolute path:
  % Data_num = fullfile(adata_path, 'data', 'SV', 'TC4_test', 'probes.csv');
end

% Old: file_num2 = append('Elena/202002/wavemaker202002 ',num2str(p-1),'.dat');
% Main directory path:
file_num2 = append('202002/wavemaker202002 ',num2str(p-1),'.dat');
% New absolute path:
% file_num2 = fullfile(adata_path, '202002', append('wavemaker202002 ', num2str(p-1), '.dat'));
Data_num2= load(file_num2);

% Main directory path old: save_name = append('FFT_',probe,'(',scheme,').png');
% New absolute directory path:
if scheme == "MMP-VP"
    % main directory one:
    save_dir = fullfile('data', 'MMP', 'TC4_test');
    % Absolute one: save_dir = fullfile(adata_path, 'data', 'MMP', 'TC4_test');
 else
    % Main directory one:
    save_dir = fullfile('data', 'SV', 'TC4_test');
    % Absolute one: save_dir = fullfile(adata_path, 'data', 'SV', 'TC4_test');
end
% Main directory one:
save_name = append('FFT_',probe,'(',scheme,').png');
% Absolute directory one: save_name = fullfile(save_dir, append('FFT_', probe, '(', scheme, ').png'));

Tend = 119.98;
H0=1;

h_exp = Data_exp(1:6000, 2);
h_num = Data_num(:,p+1)-H0;
h_num2 =Data_num2(21:119981,2);

dt_exp = Data_exp(2,1) - Data_exp(1,1);
dt_num = Data_num(2,1) - Data_num(1,1); 
dt_num2 = Data_num2(2,1) - Data_num2(1,1); 

Y_exp = fft(h_exp);
Y_num = fft(h_num);
Y_num2 = fft(h_num2);

figure(1)
s = [0:1:5999];
omg = 2*pi*s/Tend;

plot(omg(1:1000),abs(Y_exp(1:1000))*dt_exp, '-r','LineWidth',2,'DisplayName','Experimental data');
hold on
plot(omg(1:1000),abs(Y_num(1:1000))*dt_num, '-b','LineWidth',2,'DisplayName',append(scheme,' (present model)'));
plot(omg(1:1000),abs(Y_num2(1:1000))*dt_num2, ':c','LineWidth',2,'DisplayName','Gagarina et al. JCP 2014');
hold off

switch p
    case 1
        titlename = 'Probe 1, x = 10 m';
    case 2
        titlename = 'Probe 2, x = 20 m';
    case 3
        titlename = 'Probe 3, x = 40 m';
    case 4
        titlename = 'Probe 4, x = 49.5 m';
    case 5
        titlename = 'Probe 5, x = 50 m';
    case 6
        titlename = 'Probe 6, x = 54 m';
end

title(titlename,'FontSize',16)
set(gca,'XTick',[0:5:40],'FontSize',24)
set(gcf,'position',[100,400,1500,300])
axis([0 40 0 0.035]);

xlabel('\omega','FontSize',24)
ylabel('Amplitude','FontSize',24)
legend('Location','northeast','FontSize',24)
exportgraphics(gca,save_name,'Resolution',300)
