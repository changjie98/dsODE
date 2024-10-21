function [output] = DIFODE_module_bins(state,params)
output = struct;

%% preparation
dt = params.dt;
tau_r = params.tau_r;
V_bin = params.V_bin;
V_bin_min = params.V_bin_min;
V_bin_num = params.V_bin_num;
digit_num = params.digit_num;
t_end = 2;
i = 2;
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

n_n_up_mat = zeros(V_bin_num, V_bin_num);
n_n_down_mat = zeros(V_bin_num, V_bin_num);
n_n_remain_mat = zeros(V_bin_num, V_bin_num);
V_n_up_mat = zeros(V_bin_num, V_bin_num);
V_n_down_mat = zeros(V_bin_num, V_bin_num);
V_n_remain_mat = zeros(V_bin_num, V_bin_num);

%% run model
for k = find(n_n(1,:))
    %% Calculate the probability distribution of V (uniform distribution)
    V_highlim = V_bin*(k+V_bin_min);
    V_lowlim = V_bin*(k+V_bin_min-1);
    if V_n_mean(i-1,k) < V_lowlim+0.0001
        a = V_lowlim;
        b = V_lowlim+0.0001;
    elseif V_n_mean(i-1,k) <= (V_highlim+V_lowlim)/2
        a = V_lowlim;
        b = 2*V_n_mean(i-1,k)-V_lowlim;
    elseif V_n_mean(i-1,k) >= V_highlim
        a = V_highlim-0.0001;
        b = V_highlim;
    else
        a = 2*V_n_mean(i-1,k)-V_highlim;
        b = V_highlim;
    end
    
    if params.tau_m ~= 0
        a = a-a/params.tau_m*dt;
        b = b-b/params.tau_m*dt;
    end
    
    %% Neuron current in each interval
    j = 1:V_bin_num;
    mu = I_n_mean(1,k);
    sigma = sqrt(I_n_var(1,k));
    V_lim = V_bin*(j+V_bin_min);

    %% Convolution of current and voltage from the previous moment, 
    % calculate the jumps of neurons in each interval
    n_probability_accumulation = (sqrt(2./pi).*sigma.*(exp(-(b+mu-V_lim).^2./(2.*sigma.^2))-exp(-(a+mu-V_lim).^2./(2.*sigma.^2)))...
        -(a+mu-V_lim).*erf((a+mu-V_lim)./(sqrt(2).*sigma))+(b+mu-V_lim).*erf((b+mu-V_lim)./(sqrt(2).*sigma)))./(2.*(a-b));
    n_probability_accumulation = diff([n_probability_accumulation 0.5]);
    n_probability_accumulation(abs(n_probability_accumulation) < 10.^(-digit_num))=0;% 因为数值计算的关系，有时候0.5-0.5会出现一个极小的数字，要去掉
    n_probability_accumulation(isnan(n_probability_accumulation)) = 0; % 因为数值计算问题，将空值变为0
    n_n_num = round(n_n(i-1,k).*n_probability_accumulation,digit_num);
    n_n_num(n_n_num<0) = 0;
    n_n_num(n_n_num==10^(-digit_num)) = 0;
    nozero_ind = find(n_n_num); % Only intervals with non-zero number of neurons have computational significance
    V_probability_accumulation = (a.^2.*erf((-a-mu+V_lim)./(sqrt(2).*sigma))+mu.^2.*erf((-a-mu+V_lim)./(sqrt(2).*sigma))...
        +sigma.^2.*erf((-a-mu+V_lim)./(sqrt(2).*sigma))-sqrt(2./pi).*sigma.*(a+mu+V_lim).*exp(-(a+mu-V_lim).^2./(2.*sigma.^2))...
        +V_lim.^2.*erf((a+mu-V_lim)./(sqrt(2).*sigma))+2.*a.*mu.*erf((-a-mu+V_lim)./(sqrt(2).*sigma))...
        -b.^2.*erf((-b-mu+V_lim)./(sqrt(2).*sigma))-mu.^2.*erf((-b-mu+V_lim)./(sqrt(2).*sigma))...
        -sigma.^2.*erf((-b-mu+V_lim)./(sqrt(2).*sigma))+sqrt(2./pi).*sigma.*(b+mu+V_lim).*exp(-(b+mu-V_lim).^2./(2.*sigma.^2))...
        -V_lim.^2.*erf((b+mu-V_lim)./(sqrt(2).*sigma))-2.*b.*mu.*erf((-b-mu+V_lim)./(sqrt(2).*sigma)))./(4.*(a-b));
    V_probability_accumulation = diff([V_probability_accumulation (mu+(a+b)/2)/2]);
    V_probability_accumulation(abs(V_probability_accumulation) < 10.^(-digit_num))=0;
    % Due to numerical calculations, sometimes there may be a very small  
    % number between 0.5-0.5 that needs to be removed
    V_n_num = zeros(1,V_bin_num);
    V_n_num(nozero_ind) = V_probability_accumulation(nozero_ind)./n_probability_accumulation(nozero_ind);
    V_n_num(isnan(V_n_num)) = 0;
    V_n_num(n_n_num<0) = 0;

    % Splicing matrix
    n_n_up_mat(k,k+1:end) = n_n_num(k:end-1);
    n_n_down_mat(k,2:k-1) = n_n_num(1:k-2);
    n_n_remain_mat(k,k) = n_n_num(k-1);
    V_n_up_mat(k,k+1:end) = V_n_num(k:end-1);
    V_n_down_mat(k,2:k-1) = V_n_num(1:k-2);
    V_n_remain_mat(k,k) = V_n_num(k-1);

    nf_n(i) = nf_n(i) + n_n_num(end);
    
end

%% Calculate the result after jumping
nf_n(i) = nf_n(i);
fr_n(i) = nf_n(i)/dt;
ref_n(i) = ref_n(i-1) + nf_n(i);

% Calculate neurons with refractory period
if tau_r ~= 0
    dref = -ref_n(i)/tau_r + nf_n(i);
    leave_n(i) = abs(round(dref*dt,digit_num));
    ref_n(i) = ref_n(i) - leave_n(i);
else
    leave_n(i) = ref_n(i);
    ref_n(i) = 0;
end

% Calculate the number of neurons and voltage for each interval
n_n(i,:) = sum(n_n_remain_mat) + sum(n_n_down_mat) + sum(n_n_up_mat);
n_n(i,end-1) = n_n(i,end-1) - x;
V_n_mean(i,:) = sum((V_n_remain_mat.*n_n_remain_mat + V_n_up_mat.*n_n_up_mat + V_n_down_mat.*n_n_down_mat))./n_n(i,:);
% The 0 interval still needs to consider the refractory period
j = 1 - V_bin_min;
n_n(i,j) = n_n(i,j) + leave_n(i);
V_n_mean(i,j) = sum((V_n_remain_mat(:,j).*n_n_remain_mat(:,j) + V_n_up_mat(:,j).*n_n_up_mat(:,j) + V_n_down_mat(:,j).*n_n_down_mat(:,j)) + 0*leave_n(i))./n_n(i,j);

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
