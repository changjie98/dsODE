function [output] = dsODE_sigmodbins(state,params)
output = struct;

%% preparation
dt = params.dt;
tau_r = params.tau_r;
V_bin = params.V_bin;
V_bin_min = params.V_bin_min;
V_bin_num = params.V_bin_num;
V_bin_edge = V_bin*((V_bin_min):(V_bin_num+V_bin_min));
t_end = 2;
i = 1;
V_n_mean = zeros(t_end, V_bin_num);
I_n_mean = zeros(t_end, V_bin_num);
I_n_var = zeros(t_end, V_bin_num);
n_n = zeros(t_end, V_bin_num);

nf_n = zeros(t_end, 1);
fr_n = zeros(t_end, 1);
ref_n = zeros(t_end, 1);
leave_n = zeros(t_end, 1);

n_n(1,:) = state.n_n;
V_n_all = state.V_n_all;
V_n_mean(1,:) = V_n_all./n_n(1,:);
fr_n(1) = state.fr_n(1);
nf_n(1) = state.fr_n(1)*params.dt;
ref_n(1) = state.ref_n;
I_n_mean(1,:) = state.I_n_mean; 
I_n_var(1,:) = state.I_n_var;

n_up_mat = zeros(V_bin_num, V_bin_num);
n_down_mat = zeros(V_bin_num, V_bin_num);
n_remain_mat = zeros(V_bin_num, V_bin_num);
nV_up_mat = zeros(V_bin_num, V_bin_num);
nV_down_mat = zeros(V_bin_num, V_bin_num);
nV_remain_mat = zeros(V_bin_num, V_bin_num);
nf_mat = zeros(V_bin_num, 1);

%% run model
for k = find(n_n(1,:))
    mu = I_n_mean(1,k);
    mu(isnan(mu)) = 0;
    sigma2 = I_n_var(1,k);
    sigma2(isnan(sigma2)) = 0;
    p = 1./(1+exp((0.8+1./(sigma2+0.9)).*(V_bin_edge(1:end)-V_n_mean(i,k)-0.5*mu))); 
    p = abs(diff([1 p 0]));
    p(isnan(p)) = 0;
    dn = n_n(i,k).*p*dt;

    % 计算跳入每个一个区间的电流总量n*V
    V_n_now = V_n_mean(i,k) + mu*dt;
    V_n_now(isnan(V_n_now)) = 0;
    dnV = dn .* V_n_now;
    
    % Splicing matrix
    n_up_mat(k,k+1:end) = dn(k+2:end-1);
    n_down_mat(k,1:k-1) = dn(2:k);
    n_remain_mat(k,k) = n_n(i,k) - sum(dn(k+2:end)) - sum(dn(1:k));
    nf_mat(k) = dn(end);
    nV_up_mat(k,k+1:end) = dnV(k+2:end-1);
    nV_down_mat(k,1:k-1) = dnV(2:k);
    nV_remain_mat(k,k) = n_remain_mat(k,k) * V_n_now;
end

% 计算每个区间的神经元数目
n_n(2,:) = sum(n_remain_mat) + sum(n_down_mat) + sum(n_up_mat);

% 加权计算每个区间的平均V
nV_now = sum(nV_remain_mat) + sum(nV_down_mat) + sum(nV_up_mat);
V_n_mean(i+1,:) = nV_now./n_n(2,:);

%% Calculate the result after jumping
nf_n(i+1) = sum(nf_mat);
fr_n(i+1) = nf_n(i+1)/dt;
ref_n(i+1) = ref_n(i) + nf_n(i+1);

% Calculate neurons with refractory period
if tau_r ~= 0
    dref = -ref_n(i+1)/tau_r + nf_n(i+1);
    leave_n(i+1) = abs(dref*dt);
    ref_n(i+1) = ref_n(i+1) - leave_n(i+1);
else
    leave_n(i+1) = ref_n(i+1);
    ref_n(i+1) = 0;
end

% The 0 interval still needs to consider the refractory period
j = 1 - V_bin_min;
n_n(i+1,j) = n_n(i+1,j) + leave_n(i+1);
V_n_mean(i+1,j) = sum(nV_now(j) + 0*leave_n(i+1))./n_n(i+1,j);

% n_n(2,n_n(2,:)<10^-digit_num) = 0; % 去除很小的值
% V_n_mean(2,n_n(2,:)<10^-digit_num) = NaN; % 去除很小的值
output.n_n = n_n(2,:);
output.V_n_mean = V_n_mean(2,:);
output.V_n_all = V_n_mean(2,:).*n_n(2,:);
output.V_n_all(isnan(output.V_n_all)) = 0;
output.ref_n = ref_n(2);
output.nf_n = nf_n(2);
output.fr_n = fr_n(2);
output.I_n_mean = I_n_mean(2);
output.I_n_var = I_n_var(2);
end
