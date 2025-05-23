function [output] = dsODE_relu(params,I)
%% preparation
dt = params.dt;
V_bin = params.V_bin;
V_bin_min = params.V_bin_min;
V_bin_num = params.V_bin_num;
V_bin_edge = V_bin*((V_bin_min):(V_bin_num+V_bin_min));
t_end = 2;
i = 1;
n_n = zeros(t_end, V_bin_num);
nf_n = zeros(t_end, 1);
fr_n = zeros(t_end, 1);
original_size = size(I);
I = I(:); % 展平为列向量
output = zeros(1,length(I));

n_n(1,1:abs(V_bin_min)) = 1;
nf_mat = zeros(V_bin_num, 1);

%% run model
for l = 1:length(I)
    for k = find(n_n(1,:))
        a = V_bin_edge(k)+I(l);
        b = V_bin_edge(k+1)+I(l);
        p = uniform_cdf(a, b, V_bin_edge); 
        p = abs(diff([0 p 1]));
        p(isnan(p)) = 0;
        dn = n_n(i,k).*p*dt;
        nf_mat(k) = sum(dn(2-V_bin_min:end));
    end
    %% Calculate the result after jumping
    nf_n(i+1) = sum(nf_mat);
    fr_n(i+1) = nf_n(i+1)/dt;

    output(l) = fr_n(2);
end

output = reshape(output, original_size);


function cdf_values = uniform_cdf(a, b, edges)
    % 计算均匀分布U(a, b)在edges各点处的CDF值
    % 参数检查
    if b <= a
        error('错误：b必须大于a。');
    end
    % 计算CDF值
    cdf_values = (edges - a) / (b - a);
    cdf_values = max(0, min(1, cdf_values));
end
end
