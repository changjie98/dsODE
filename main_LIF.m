tic;
params = struct;
params.Ex_Poisson_lambda = 7;
params.M = 100;
params.Mr = 66;
k = 100/100;
params.ne = 300/k;
params.ni = 100/k;
params.dt = 0.1;
params.duration_time = 1000;
params.tau_adapt = 0;
params.tau_m = 0;
params.tau_ee = 1.4;
params.tau_ie = 1.2;
params.tau_i = 4.5;
params.tau_r = 4;
params.p_ee = 0.8;
params.p_ie = 0.8; 
params.p_ei = 0.8; 
params.p_ii = 0.8; 
params.s_ee     = 0.95*k;
params.s_ie     = 1.25*k;
params.s_ei     = 2.71*k;
params.s_ii     = 2.45*k;

% res_lif = run_LIF_model(params);
res_lif = run_LIFSDE_model(params);
    

toc
