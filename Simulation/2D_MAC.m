clear;
clc; 
clear fig;
close all


R = .4;
M_LENGTH = 32/R;
T_LENGTH = 32/R;

P = .1:.1:10;
SIGMA_N_SQUARD = 1;


X = 5; % number of rows 
Y = 5; % number of columns

ALPHA = 0.22;



function p = BER_before_decoding(gamma)
    p = qfunc(sqrt(2*gamma));
end

function gamma = SNR(P,sigma_n_squared)
    gamma = P./sigma_n_squared;
end

function C = capacitY(p)
    C = 1+p.*log2(p) + (1-p).*log2(1-p);
end

function Pe = error_rate(length, R, p, C)
    Pe = qfunc(sqrt(length ./ (p .* (1 - p))) .* ((C - R) / log2((1 - p) ./ p)));
end

function throuput = auth_throughput(R, AER, Pe_m, m_length, t_length)
    throuput = R * (1-AER).*(1-Pe_m).*m_length /(m_length + t_length);
end



%%%%%%%% Tradiotional scheme (0) %%%%%%%%%%
function A_0 = A_rate_0(Pe_m, Pe_t)
    A_0 = Pe_t;
end

function AER_0 = auth_error_rate_0(Pe_m, Pe_t)
    AER_0 = Pe_t;
end


gamma_0 = SNR(P, SIGMA_N_SQUARD);
p_0 = BER_before_decoding (gamma_0);
C_0 = capacitY(p_0);

message_loss_rate_0 = error_rate(M_LENGTH, R, p_0, C_0);
tag_loss_rate_0 = error_rate(T_LENGTH, R, p_0, C_0);

A_0 = A_rate_0(message_loss_rate_0,tag_loss_rate_0);
AER_0 = auth_error_rate_0(message_loss_rate_0,tag_loss_rate_0);
Throuput_0 = auth_throughput(R, A_0,message_loss_rate_0, M_LENGTH, T_LENGTH);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%% 1D MAC scheme (1) %%%%%%%%%%

function A_1 = A_rate_1(Pe_m, Pe_t, X)
    A_1 = 1-((1-Pe_m).^(X-1)).*(1-Pe_t);
end

function AER_1 = auth_error_rate_1(Pe_m, Pe_t, X)
    AER_1 = 1-((1-Pe_m).^(X)).*(1-Pe_t);
end

P_1 = ((X*M_LENGTH + X*T_LENGTH)/ (X*M_LENGTH + T_LENGTH)).*P;
gamma_1 = SNR(P_1, SIGMA_N_SQUARD);
p_1 = BER_before_decoding (gamma_1);
C_1 = capacitY(p_1);

message_loss_rate_1 = error_rate(M_LENGTH, R, p_1, C_1);
tag_loss_rate_1     = error_rate(T_LENGTH, R, p_1, C_1);

A_1 = A_rate_1(message_loss_rate_1,tag_loss_rate_1, X);
AER_1 = auth_error_rate_1(message_loss_rate_1,tag_loss_rate_1, X);
Throuput_1 = auth_throughput(R, A_1, message_loss_rate_1, X*M_LENGTH, 1*T_LENGTH);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%% 2D MAC scheme (2) %%%%%%%%%%

function A_2 = A_rate_2(Pe_m, Pe_t, X, Y)
    A_2 = (1-((1-Pe_m).^(Y-1)).*(1-Pe_t)).*(1-((1-Pe_m).^(X-1)).*(1-Pe_t));
end

function AER_2 = auth_error_rate_2(Pe_m, Pe_t, X, Y)
    AER_2 = (1-Pe_m).*A_rate_2(Pe_m, Pe_t, X,Y) + Pe_m;
end



P_2 = ((X*Y*M_LENGTH + X*Y*T_LENGTH)/ (X*Y*M_LENGTH + (X+Y)*T_LENGTH)).*P;
gamma_2 = SNR(P_2, SIGMA_N_SQUARD);
p_2 = BER_before_decoding (gamma_2);
C_2 = capacitY(p_2);

message_loss_rate_2 = error_rate(M_LENGTH, R, p_2, C_2);
tag_loss_rate_2     = error_rate(T_LENGTH, R, p_2, C_2);

A_2 = auth_error_rate_2(message_loss_rate_2,tag_loss_rate_2, X, Y);
AER_2 = auth_error_rate_2(message_loss_rate_2, tag_loss_rate_2, X, Y);
Throuput_2 = auth_throughput(R, A_2, message_loss_rate_2, (X*Y)*M_LENGTH, (X+Y)*T_LENGTH);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









%%%%%%%% 2D MAC scheme (3) %%%%%%%%%%


function gamma_m = SINR_M (P, sigma_n_squared, alpha)
    gamma_m = ((1-alpha)*P) ./ (alpha*P + sigma_n_squared);
end

function gamma_t = SINR_T (P, sigma_n_squared, alpha)
    gamma_t = (alpha.*P)./((1-alpha).*P + sigma_n_squared);
end

function A_3 = A_rate_3(Pe_m_prime, Pe_t_prime, Pe_t, X,Y)
    A_3 = (1-((1-Pe_m_prime).^(Y-1)).*(1-Pe_t_prime)).*(1-((1-Pe_m_prime).^(X-1)).*(1-Pe_t));
end

function AER_3 = auth_error_rate_3(Pe_m_prime, Pe_t_prime, Pe_t, X,Y)
    AER_3 = (1-Pe_m_prime).*A_rate_3(Pe_m_prime, Pe_t_prime, Pe_t, X,Y) + Pe_m_prime;
end

P_3 = ((X*M_LENGTH + X*T_LENGTH)/ (X*M_LENGTH + T_LENGTH)).*P;

gamma_t = SNR(P_3, SIGMA_N_SQUARD);
gamma_m_prime = SINR_M (P_3, SIGMA_N_SQUARD, ALPHA);
gamma_t_prime = SINR_T (P_3, SIGMA_N_SQUARD, ALPHA);


p_t       = BER_before_decoding (gamma_t);
p_t_prime = BER_before_decoding (gamma_t_prime);
p_m_prime = BER_before_decoding (gamma_m_prime);

C_t       = capacitY (p_t);
C_t_prime = capacitY (p_t_prime);
C_m_prime = capacitY (p_m_prime);


tag_loss_rate_3           = error_rate (T_LENGTH, R, p_t, C_t);
tag_loss_rate_3_prime     = error_rate (Y*M_LENGTH, (T_LENGTH/(Y*M_LENGTH))*R, p_t_prime, C_t_prime);
message_loss_rate_3_prime = error_rate (M_LENGTH, R, p_m_prime, C_m_prime);



A_3 = A_rate_3(message_loss_rate_3_prime, tag_loss_rate_3_prime, tag_loss_rate_3 , X, Y);
AER_3 = auth_error_rate_3(message_loss_rate_3_prime, tag_loss_rate_3_prime, tag_loss_rate_3 , X, Y);
Throuput_3 = auth_throughput(R, A_3,message_loss_rate_3_prime, X*M_LENGTH, 1*T_LENGTH);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 





snr_dB = 10.*log10(gamma_0);
 
% Define colors
bgColor = [240, 240, 240] / 255; % Light gray for plot area
borderColor = [1, 1, 1];         % White for figure border

% New color palette for lines
lineColors = [
    0, 0.4470, 0.7410;   % blue
    0.8500, 0.3250, 0.0980; % orange
    0.9290, 0.6940, 0.1250; % yellow
    0.4940, 0.1840, 0.5560; % purple
];


% Plot for AER vs SNR

figure


plot(snr_dB, AER_0, 'Color', lineColors(1, :), 'LineWidth', 5.5); hold on;
plot(snr_dB, AER_1, 'Color', lineColors(2, :), 'LineWidth', 5.5);
plot(snr_dB, AER_2, 'Color', lineColors(3, :), 'LineWidth', 5.5);
plot(snr_dB, AER_3, 'Color', lineColors(4, :), 'LineWidth', 5.5);
hold off;

legend("Trad.", "1D MAC", "2D MAC", "2D MAC with SC", 'FontSize', 28, 'Location', 'northeast');
ylabel("AER", 'FontSize', 33, 'FontWeight', 'bold');
xlabel("SNR (dB)", 'FontSize', 33, 'FontWeight', 'bold');
% title("AER vs SNR", 'FontSize', 20, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 33, 'LineWidth',1.5, 'XColor', 'k', 'YColor', 'k');

% Plot for Auth Throughput vs SNR
figure


plot(snr_dB, Throuput_0, 'Color', lineColors(1, :), 'LineWidth', 5.5); hold on;
plot(snr_dB, Throuput_1, 'Color', lineColors(2, :), 'LineWidth', 5.5);
plot(snr_dB, Throuput_2, 'Color', lineColors(3, :), 'LineWidth', 5.5);
plot(snr_dB, Throuput_3, 'Color', lineColors(4, :), 'LineWidth', 5.5);
hold off;

legend("Trad.", "1D MAC", "2D MAC", "2D MAC with SC", 'FontSize', 28, 'Location', 'northwest');
ylabel("Auth Throughput", 'FontSize', 33, 'FontWeight', 'bold');
xlabel("SNR (dB)", 'FontSize', 33, 'FontWeight', 'bold');
% title("Auth Throughput vs SNR", 'FontSize', 33, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 33, 'LineWidth', 1.5, 'XColor', 'k', 'YColor', 'k');




%%%%%%%%%%%%%%%% optimizing the alpha %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
