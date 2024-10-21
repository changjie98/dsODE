function [res] = run_LIFSDE_model(params,start)
res = struct;
flag = true;
if nargin > 2
    error('too many variables！');
elseif nargin == 1
    flag = false; 
end
%% preparation
dt = params.dt;
ne = params.ne;
ni = params.ni;
pee = params.p_ee; 
pie = params.p_ie; 
pei = params.p_ei; 
pii = params.p_ii; 
tau_ee = params.tau_ee;
tau_ie = params.tau_ie;
tau_i = params.tau_i;
tau_r = params.tau_r;
t_end = params.duration_time/dt;
V_e = zeros(t_end, ne);
V_i = zeros(t_end, ni);
fr_e = zeros(t_end, 1);
fr_i = zeros(t_end, 1);
ref_e = zeros(t_end, ne);
ref_i = zeros(t_end, ni);
I_e_mean = zeros(t_end, ne);
I_e_var = zeros(t_end, ne);
I_i_mean = zeros(t_end, ni);
I_i_var = zeros(t_end, ni);
H_ee_mean = zeros(t_end, 1);
H_ei_mean = zeros(t_end, 1);
H_ie_mean = zeros(t_end, 1);
H_ii_mean = zeros(t_end, 1);
H_ee_var = zeros(t_end, 1);
H_ei_var = zeros(t_end, 1);
H_ie_var = zeros(t_end, 1);
H_ii_var = zeros(t_end, 1);
ft = zeros(t_end, ne+ni);
I_ee = zeros(t_end, ne);
I_ie = zeros(t_end, ni);
I_ei = zeros(t_end, ne);
I_ii = zeros(t_end, ni);
I_e_list = zeros(t_end, ne);
I_i_list = zeros(t_end, ni);
pd = makedist('Normal',0,sqrt(dt));
if flag
    V_e(1,:) = start.V_e;
    V_i(1,:) = start.V_i;
    H_ee_mean(1) = start.H_ee_mean;
    H_ie_mean(1) = start.H_ie_mean;
    H_ei_mean(1) = start.H_ei_mean;
    H_ii_mean(1) = start.H_ii_mean; 
    H_ee_var(1) = start.H_ee_var;
    H_ie_var(1) = start.H_ie_var;
    H_ei_var(1) = start.H_ei_var;
    H_ii_var(1) = start.H_ii_var; 
    fr_e(1) = start.fr_e;
    fr_i(1) = start.fr_i;
    ref_e(1,:) = start.ref_e;
    ref_i(1,:) = start.ref_i;
end

%% run model
for i = 2:t_end % Calculate from 'this moment'
    % Calculate the distribution parameters of the current at the previous moment
    dH_ee_mean = -H_ee_mean(i-1)/tau_ee + fr_e(i-1)*pee; % mean
    dH_ee_var = -H_ee_var(i-1)*2/(tau_ee) + H_ee_mean(i-1)/tau_ee + fr_e(i-1)*pee*(1-pee); % var
    H_ee_mean(i) = H_ee_mean(i-1) + dH_ee_mean*dt;
    H_ee_var(i) = H_ee_var(i-1) + dH_ee_var*dt;

    dH_ei_mean = -H_ei_mean(i-1)/tau_i + fr_i(i-1)*pei; % mean
    dH_ei_var = -H_ei_var(i-1)*2/(tau_i) + H_ei_mean(i-1)/tau_i + fr_i(i-1)*pei*(1-pei); % var
    H_ei_mean(i) = H_ei_mean(i-1) + dH_ei_mean*dt;
    H_ei_var(i) = H_ei_var(i-1) + dH_ei_var*dt;

    % Calculate current and probability distribution
    J_eex = 1;
    J_ee = params.s_ee;
%     J_ei = params.s_ei;   
    J_ei = (V_e(i-1,:)+params.Mr)*params.s_ei/(params.M+params.Mr);    
    
    if params.tau_m == 0
        dW = random(pd,1,ne);
        I_e_mean(i,:) = J_ee*H_ee_mean(i)/params.tau_ee - J_ei.*H_ei_mean(i)/params.tau_i + J_eex * params.Ex_Poisson_lambda; % 注意这里有/dt有*dt，代表的含义不一样，所以分开写
        I_e_var(i,:) = J_ee.^2.*H_ee_var(i)/params.tau_ee^2 + J_ei.^2.*H_ei_var(i)/params.tau_i^2 + J_eex *params.Ex_Poisson_lambda;
        I_e = I_e_mean(i,:) .* dt + sqrt(I_e_var(i)) .* dW;
    else
        dW = random(pd,1,ne);
        I_e_mean(i,:) = J_ee*H_ee_mean(i)/params.tau_ee - J_ei.*H_ei_mean(i)/params.tau_i + J_eex * params.Ex_Poisson_lambda/params.tau_m; % 注意这里有/dt有*dt，代表的含义不一样，所以分开写
        I_e_var(i,:) = J_ee.^2.*H_ee_var(i)/params.tau_ee^2 + J_ei.^2.*H_ei_var(i)/params.tau_i^2 + J_eex * params.Ex_Poisson_lambda/params.tau_m^2;
        I_e = I_e_mean(i,:) .* dt + sqrt(I_e_var(i)) .* dW;
        I_e_leak = V_e(i-1,:)./params.tau_m;
        I_e = I_e - I_e_leak*dt;
    end
    I_e_list(i,:) = I_e;
    
    % Calculate the distribution parameters of the current at the previous moment
    dH_ie_mean = -H_ie_mean(i-1)/tau_ie + fr_e(i-1)*pie; % mean
    dH_ie_var = -H_ie_var(i-1)*2/(tau_ie) + H_ie_mean(i-1)/tau_ie + fr_e(i-1)*pie*(1-pie); % var
    H_ie_mean(i) = H_ie_mean(i-1) + dH_ie_mean*dt;
    H_ie_var(i) = H_ie_var(i-1) + dH_ie_var*dt;

    dH_ii_mean = -H_ii_mean(i-1)/tau_i + fr_i(i-1)*pii; % mean
    dH_ii_var = -H_ii_var(i-1)*2/(tau_i) + H_ii_mean(i-1)/tau_i + fr_i(i-1)*pii*(1-pii); % var
    H_ii_mean(i) = H_ii_mean(i-1) + dH_ii_mean*dt;
    H_ii_var(i) = H_ii_var(i-1) + dH_ii_var*dt;

    % Calculate current and probability distribution
    J_iex = 1;
    J_ie = params.s_ie;
%     J_ii = params.s_ii;   
    J_ii = (V_i(i-1,:)+params.Mr)*params.s_ii/(params.M+params.Mr);
    
    if params.tau_m == 0
        I_i_mean(i,:) = J_ie*H_ie_mean(i)/params.tau_ie - J_ii*H_ii_mean(i)/params.tau_i + J_iex * params.Ex_Poisson_lambda; % 注意这里有/dt表的含义
        I_i_var(i,:) = J_ie.^2*H_ie_var(i)/params.tau_ie^2 + J_ii.^2*H_ii_var(i)/params.tau_i^2 + J_iex * params.Ex_Poisson_lambda;
        dW = random(pd,1,ni);
        I_i = I_i_mean(i,:) * dt + sqrt(I_i_var(i,:)) .* dW;
    else
        I_i_mean(i,:) = J_ie*H_ie_mean(i)/params.tau_ie - J_ii*H_ii_mean(i)/params.tau_i + J_iex * params.Ex_Poisson_lambda./params.tau_m; % 注意这里有/dt表的含义
        I_i_var(i,:) = J_ie.^2*H_ie_var(i)/params.tau_ie^2 + J_ii.^2*H_ii_var(i)/params.tau_i^2 + J_iex * params.Ex_Poisson_lambda./params.tau_m^2;
        dW = random(pd,1,ni);
        I_i = I_i_mean(i,:) * dt + sqrt(I_i_var(i,:)) .* dW;
        I_i_leak = V_i(i-1,:)./params.tau_m;
        I_i = I_i - I_i_leak*dt;
    end
    I_i_list(i,:) = I_i;
    
    %% The voltage at the previous moment is obtained by adding current to the voltage. 
    %The voltage above the threshold is reset to zero and added to the nf array
    e_index = find(~ref_e(i,:));
    i_index = find(~ref_i(i,:));
    V_e(i,e_index) =  I_e(e_index) + V_e(i-1,e_index);
    V_i(i,i_index) =  I_i(i_index) + V_i(i-1,i_index);
    fr_e(i) = sum(V_e(i,:) > params.M)/dt;
    fr_i(i) = sum(V_i(i,:) > params.M)/dt;
    ft(i,:)=([V_e(i,:) > params.M V_i(i,:) > params.M]);

    ref_e(i,V_e(i,:) > params.M) = 1;
    ref_i(i,V_i(i,:) > params.M) = 1;
    
    V_e(i,V_e(i,:) > params.M) = 0;
    V_i(i,V_i(i,:) > params.M) = 0;
    
    %% Calculate the ref at this moment
    if params.tau_r ~= 0
        ref_e_consume = random('Poisson',ref_e(i,:)./params.tau_r*dt);
        ref_i_consume = random('Poisson',ref_i(i,:)./params.tau_r*dt);    
%         ref_e_consume = ref_e(i,:).*(1-exp(-dt/params.tau_r));
%         ref_i_consume = ref_i(i,:).*(1-exp(-dt/params.tau_r));
        ref_e(i+1,:) = ref_e(i,:) - ref_e_consume;    
        ref_i(i+1,:) = ref_i(i,:) - ref_i_consume;
        ref_e(i+1,(ref_e(i+1,:)<0.37)) = 0;
        ref_i(i+1,(ref_i(i+1,:)<0.37)) = 0;
    else
        ref_e(i,V_e(i,:) > params.M) = 0;
        ref_i(i,V_i(i,:) > params.M) = 0;
    end
end


res.V_e = V_e;
res.V_i = V_i;
res.H_ie_mean = H_ie_mean;
res.H_ii_mean = H_ii_mean;
res.H_ee_mean = H_ee_mean;
res.H_ei_mean = H_ei_mean;
res.H_ie_var = H_ie_var;
res.H_ii_var = H_ii_var;
res.H_ee_var = H_ee_var;
res.H_ei_var = H_ei_var;
res.fr_e = fr_e;
res.fr_i = fr_i;
res.ft = ft;
res.ref_e = ref_e(2:end,:);
res.ref_i = ref_i(2:end,:);
res.I_e = I_e_list;
res.I_i = I_i_list;
res.I_e_var = I_e_var;
res.I_i_var = I_i_var;
res.t = dt:dt:params.duration_time;

end
