tic
%% 模型
res_DIFODE = struct;
params = struct;
%% 设置参数
params.J_ex = 7;
params.M = 100;
params.Mr = 66;
k = 100/100;
params.ne = 300/k;
params.ni = 100/k;
params.dt = 0.1;
params.duration_time = 1000;
params.tau_ee = 1.4;
params.tau_ie = 1.2;
params.tau_i = 4.5;
params.tau_r = 4;
params.tau_m = 0;
params.p_ee = 0.8; 
params.p_ie = 0.8; 
params.p_ei  = 0.8; 
params.p_ii = 0.8; 
params.s_ee     = 0.95*k;
params.s_ie     = 1.25*k;
params.s_ei     = 2.71*k;
params.s_ii     = 2.45*k;

params.V_bin = 4; % Voltage interval size
params.V_bin_min = -10; % The minimum number of voltage intervals
params.V_bin_num = params.M /params.V_bin - params.V_bin_min;
params.digit_num = 10; % Round to the nearest decimal place in numerical calculations
params.t_end = params.duration_time/params.dt;

%% Initialize several variables
% V all bin 、Neuron number in each bin、fr、ref、H（mean、var）、I（mean、var）
V_e_all = zeros(params.t_end, params.V_bin_num);
V_e_mean = zeros(params.t_end, params.V_bin_num);
n_e = zeros(params.t_end, params.V_bin_num);

n_e(1,1-params.V_bin_min) = params.ne;
V_e_all(1,1-params.V_bin_min) = 0.0001; % If no initial state is given by default, set the total neuron voltage to 0.0001
V_e_mean(1,:) = V_e_all(1,:)./n_e(1,:);
fr_e = zeros(params.t_end, 1);
ref_e = zeros(params.t_end, 1);
H_ee_mean = zeros(params.t_end, 1);
H_ei_mean = zeros(params.t_end, 1);
I_ee_mean_list = zeros(params.t_end, 1);
I_ei_mean_list = zeros(params.t_end, 1);
I_ee_var_list = zeros(params.t_end, 1);
I_ei_var_list = zeros(params.t_end, params.V_bin_num);
H_ee_var = zeros(params.t_end, 1);
H_ei_var = zeros(params.t_end, 1);
I_e_mean = zeros(params.t_end, params.V_bin_num);
I_e_var = zeros(params.t_end, params.V_bin_num);

V_i_all = zeros(params.t_end, params.V_bin_num);
V_i_mean = zeros(params.t_end, params.V_bin_num);
n_i = zeros(params.t_end, params.V_bin_num);
V_i_all(1,1-params.V_bin_min) = 0.0001;
n_i(1,1-params.V_bin_min) = params.ni;
V_i_mean(1,:) = V_i_all(1,:)./n_i(1,:);
fr_i = zeros(params.t_end, 1);
ref_i = zeros(params.t_end, 1);
H_ie_mean = zeros(params.t_end, 1);
H_ii_mean = zeros(params.t_end, 1);
I_ie_mean_list = zeros(params.t_end, 1);
I_ii_mean_list = zeros(params.t_end, 1);
H_ie_var = zeros(params.t_end, 1);
H_ii_var = zeros(params.t_end, 1);
I_i_mean = zeros(params.t_end, params.V_bin_num);
I_i_var = zeros(params.t_end, params.V_bin_num);


%% The input can be adjusted to various forms, and the simplest stable
%  Poisson input is used here, where the mean equals the variance
I_eex_mean = zeros(params.t_end,params.V_bin_num) + params.J_ex*params.dt;
I_eex_var = zeros(params.t_end,params.V_bin_num) + params.J_ex*params.dt;
I_iex_mean = zeros(params.t_end,params.V_bin_num) + params.J_ex*params.dt;
I_iex_var = zeros(params.t_end,params.V_bin_num) + params.J_ex*params.dt;
if params.tau_m ~= 0
    I_eex_mean = zeros(params.t_end,params.V_bin_num) + params.J_ex/params.tau_m*params.dt;
    I_eex_var = zeros(params.t_end,params.V_bin_num) + params.J_ex/params.tau_m^2*params.dt;
    I_iex_mean = zeros(params.t_end,params.V_bin_num) + params.J_ex/params.tau_m*params.dt;
    I_iex_var = zeros(params.t_end,params.V_bin_num) + params.J_ex/params.tau_m^2*params.dt;
end

%% Used with get_start. m, if there is a special initial state, adjust havestart=1
havestart = 0;
if havestart
    H_ee_mean(1) = start.H_ee_mean;
    H_ei_mean(1) = start.H_ei_mean;
    H_ie_mean(1) = start.H_ie_mean;
    H_ii_mean(1) = start.H_ii_mean;
    H_ee_var(1) = start.H_ee_var;
    H_ei_var(1) = start.H_ei_var;
    H_ie_var(1) = start.H_ie_var;
    H_ii_var(1) = start.H_ii_var;
    
    n_e(1,:) = start.n_e;
    n_i(1,:) = start.n_i;
    V_e_all = start.V_e_all;
    V_i_all = start.V_i_all;
    V_e_mean(1,:) = V_e_all./n_e(1,:);
    V_i_mean(1,:) = V_i_all./n_i(1,:);
    fr_e(1) = start.fr_e(1);
    fr_i(1) = start.fr_i(1);
    ref_e(1) = start.ref_e;
    ref_i(1) = start.ref_i;
end

for i = 2:params.t_end
    %% Input various parameters from the previous moment into the module to calculate the changes
    E_state.V_n_all = V_e_all(i-1,:);
    E_state.n_n = n_e(i-1,:);
    E_state.fr_n = fr_e(i-1);
    E_state.ref_n = ref_e(i-1);    
    I_ee_mean = params.dt*params.s_ee*H_ee_mean(i-1)/params.tau_ee;
    I_ei_mean = params.dt*params.s_ei*H_ei_mean(i-1)/params.tau_i*(V_e_mean(i-1,:)+params.Mr)/(params.M+params.Mr);
%     I_ei_mean = params.dt*params.s_ei*H_ei_mean(i-1)/params.tau_i;
    I_ee_mean_list(i) = I_ee_mean;
    I_ei_mean_list(i,:) = sum(I_ei_mean.*n_e(i-1,:)./sum(n_e(i-1,:),2),2,'omitnan');
    I_ee_var = params.dt.*params.s_ee^2.*H_ee_var(i-1)./params.tau_ee^2;
    I_ei_var = params.dt.*params.s_ei^2.*H_ei_var(i-1)./params.tau_i.^2.*(V_e_mean(i-1,:)+params.Mr).^2./(params.M+params.Mr).^2;

    I_ee_var_list(i) = I_ee_var;
    I_ei_var_list(i,:) = I_ei_var;
    I_e_mean(i,:) = I_eex_mean(i-1,:) + I_ee_mean - I_ei_mean;
    I_e_var(i,:) = I_eex_var(i-1,:) + I_ee_var + I_ei_var;
    
    % You can choose whether to add random error to the current
%     x = random('Normal',0,sqrt(I_e_var(i,:)/params.ne));
%     y = random('Normal',0,sqrt(2*(I_e_var(i,:)).^2/(params.ne-1)));
%     I_e_mean(i,:) = I_e_mean(i,:) + x;
%     I_e_var(i,:) = I_e_var(i,:) + y;

    I_e_var(i,I_e_var(i,:)<0) = 0;
    E_state.I_n_mean = I_e_mean(i,:);
    E_state.I_n_var = I_e_var(i,:);
    E_output = DIFODE_module_bins(E_state,params);
    % You can choose whether to add random error to the current
%     x = random('Normal',0,sqrt(E_output.fr_n));
%     E_output.fr_n = E_output.fr_n+x;
    
    
    I_state.V_n_all = V_i_all(i-1,:);
    I_state.n_n = n_i(i-1,:);
    I_state.fr_n = fr_i(i-1);
    I_state.ref_n = ref_i(i-1);    
    I_ie_mean = params.dt*params.s_ie*H_ie_mean(i-1)/params.tau_ie;
    I_ii_mean = params.dt*params.s_ii*H_ii_mean(i-1)/params.tau_i*(V_i_mean(i-1,:)+params.Mr)/(params.M+params.Mr);
%     I_ii_mean = params.dt*params.s_ii*H_ii_mean(i-1)/params.tau_i;
    I_ie_mean_list(i,:) = I_ie_mean;
    I_ii_mean_list(i,:) = sum(I_ii_mean.*n_i(i-1,:)./sum(n_i(i-1,:),2),2,'omitnan');
    I_ie_var = params.dt.*params.s_ie.^2.*H_ie_var(i-1)./params.tau_ie.^2 ;
    I_ii_var = params.dt.*params.s_ii.^2.*H_ii_var(i-1)./params.tau_i.^2.*(V_i_mean(i-1,:)+params.Mr).^2./(params.M+params.Mr).^2;
    I_i_mean(i,:) = I_iex_mean(i-1,:) + (I_ie_mean - I_ii_mean);
    I_i_var(i,:) = I_iex_var(i-1,:) + (I_ie_var + I_ii_var);
    
    % You can choose whether to add random error to the current
%     x = random('Normal',0,sqrt(I_i_var(i,:)/params.ni));
%     y = random('Normal',0,sqrt(2*(I_i_var(i,:)).^2/(params.ni-1)));
%     I_i_mean(i,:) = I_i_mean(i,:) + x;
%     I_i_var(i,:) = I_i_var(i,:) + y;

    I_i_var(i,I_i_var(i,:)<0) = 0;
    I_state.I_n_mean = I_i_mean(i,:);
    I_state.I_n_var = I_i_var(i,:);
    I_output = DIFODE_module_bins(I_state,params);
    % You can choose whether to add random error to the current
%     x = random('Normal',0,sqrt(I_output.fr_n));
%     I_output.fr_n = I_output.fr_n+x;
   
    %% Calculate the distribution parameters of current at this moment
    dH_ee_mean = -H_ee_mean(i-1)/params.tau_ee + fr_e(i-1)*params.p_ee; % mean
    dH_ee_var = -H_ee_var(i-1)*2/(params.tau_ee) + H_ee_mean(i-1)/params.tau_ee + fr_e(i-1)*params.p_ee*(1-params.p_ee); % var
    H_ee_mean(i) = H_ee_mean(i-1) + dH_ee_mean*params.dt;
    H_ee_var(i) = H_ee_var(i-1) + dH_ee_var*params.dt;
    dH_ei_mean = -H_ei_mean(i-1)/params.tau_i + fr_i(i-1)*params.p_ei; % mean
    dH_ei_var = -H_ei_var(i-1)*2/(params.tau_i) + H_ei_mean(i-1)/params.tau_i + fr_i(i-1)*params.p_ei*(1-params.p_ei); % var
    H_ei_mean(i) = H_ei_mean(i-1) + dH_ei_mean*params.dt;
    H_ei_var(i) = H_ei_var(i-1) + dH_ei_var*params.dt;

    dH_ie_mean = -H_ie_mean(i-1)/params.tau_ie + fr_e(i-1)*params.p_ie; % mean
    dH_ie_var = -H_ie_var(i-1)*2/(params.tau_ie) + H_ie_mean(i-1)/params.tau_ie + fr_e(i-1)*params.p_ie*(1-params.p_ie); % var
    H_ie_mean(i) = H_ie_mean(i-1) + dH_ie_mean*params.dt;
    H_ie_var(i) = H_ie_var(i-1) + dH_ie_var*params.dt;
    dH_ii_mean = -H_ii_mean(i-1)/params.tau_i + fr_i(i-1)*params.p_ii; % mean
    dH_ii_var = -H_ii_var(i-1)*2/(params.tau_i) + H_ii_mean(i-1)/params.tau_i + fr_i(i-1)*params.p_ii*(1-params.p_ii); % var
    H_ii_mean(i) = H_ii_mean(i-1) + dH_ii_mean*params.dt;
    H_ii_var(i) = H_ii_var(i-1) + dH_ii_var*params.dt;

    
    %% Change output to input
    V_e_all(i,:) = E_output.V_n_all;
    V_e_mean(i,:) = E_output.V_n_mean;
    n_e(i,:) = E_output.n_n;
    fr_e(i) = E_output.fr_n;
    ref_e(i) = E_output.ref_n;
    
    V_i_all(i,:) = I_output.V_n_all;
    V_i_mean(i,:) = I_output.V_n_mean;
    n_i(i,:) = I_output.n_n;
    fr_i(i) = I_output.fr_n;
    ref_i(i) = I_output.ref_n;
end

res_DIFODE.n_e = n_e;
res_DIFODE.V_e_mean = V_e_mean;
res_DIFODE.V_e_all = V_e_all;
res_DIFODE.ref_e = ref_e;
res_DIFODE.H_ee_mean = H_ee_mean;
res_DIFODE.H_ei_mean = H_ei_mean;
res_DIFODE.I_ee_mean = I_ee_mean_list;
res_DIFODE.I_ee_var = I_ee_var_list;
res_DIFODE.I_ei_var = I_ei_var_list;
res_DIFODE.I_ei_mean = I_ei_mean_list;
res_DIFODE.H_ee_var = H_ee_var;
res_DIFODE.H_ei_var = H_ei_var;
res_DIFODE.fr_e = fr_e;
res_DIFODE.I_eex_mean = I_eex_mean;
res_DIFODE.I_e_mean = I_e_mean;
res_DIFODE.I_e_var = I_e_var;

res_DIFODE.fr_i = fr_i;
res_DIFODE.n_i = n_i;
res_DIFODE.V_i_mean = V_i_mean;
res_DIFODE.V_i_all = V_i_all;
res_DIFODE.ref_i = ref_i;
res_DIFODE.H_ie_mean = H_ie_mean;
res_DIFODE.H_ii_mean = H_ii_mean;
res_DIFODE.I_ie_mean = I_ie_mean_list;
res_DIFODE.I_ii_mean = I_ii_mean_list;
res_DIFODE.H_ie_var = H_ie_var;
res_DIFODE.H_ii_var = H_ii_var;
res_DIFODE.I_iex_mean = I_iex_mean;
res_DIFODE.I_i_mean = I_i_mean;
res_DIFODE.I_i_var = I_i_var;

clear n_e
clear V_e_all
clear V_e_mean
clear ref_e
clear H_ee_mean
clear H_ei_mean
clear H_ee_var
clear H_ei_var
clear dH_ee_mean
clear dH_ei_mean
clear dH_ee_var
clear dH_ei_var
clear I_ee_mean
clear I_ei_mean
clear I_ee_mean_list
clear I_ei_mean_list
clear I_ee_var
clear I_ei_var
clear I_e_mean
clear fr_e
clear I_e_var
clear E_output
clear I_eex
clear I_e_leak
clear I_ee_var_list
clear I_eex_mean
clear I_eex_var
clear I_ei_var_list

clear n_i
clear V_i_all
clear V_i_mean
clear ref_i
clear H_ie_mean
clear H_ii_mean
clear H_ie_mean_list
clear H_ii_mean_list
clear H_ie_var
clear H_ii_var
clear dH_ie_mean
clear dH_ii_mean
clear dH_ie_var
clear dH_ii_var
clear I_ie_mean
clear I_ii_mean
clear I_ie_var
clear I_ii_var
clear I_i_mean
clear fr_i
clear I_i_var
clear I_output
clear I_iex
clear I_i_leak
clear I_ie_var_list
clear I_iex_mean
clear I_iex_var
clear I_ii_var_list



clear i
toc
