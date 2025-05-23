% 主程序部分
% 加载转换后的权重（已调整维度顺序）
weights = struct();

% 加载卷积层1权重并处理偏置
conv1_weight = load('weights\conv1_weight.mat');
weights.conv1.weight = conv1_weight.conv1_weight; 
if exist('conv1_bias.mat', 'file')
    conv1_bias = load('conv1_bias.mat');
    weights.conv1.bias = conv1_bias.conv1_bias;
else
    % 根据卷积核输出通道数生成全零偏置（假设权重维度为[out_ch,in_ch,H,W]）
    out_channels = size(weights.conv1.weight, 1); 
    weights.conv1.bias = zeros(out_channels, 1); 
end

% 加载卷积层2权重并处理偏置
conv2_weight = load('weights\conv2_weight.mat');
weights.conv2.weight = conv2_weight.conv2_weight;
if exist('conv2_bias.mat', 'file')
    conv2_bias = load('conv2_bias.mat');
    weights.conv2.bias = conv2_bias.conv2_bias;
else
    out_channels = size(weights.conv2.weight, 1);
    weights.conv2.bias = zeros(out_channels, 1);
end

% 加载全连接层1权重并处理偏置
linear1_weight = load('weights\linear1_weight.mat');
weights.linear1.weight = linear1_weight.linear1_weight;
if exist('linear1_bias.mat', 'file')
    linear1_bias = load('linear1_bias.mat');
    weights.linear1.bias = linear1_bias.linear1_bias;
else
    % 根据全连接层输出维度生成全零偏置（假设权重维度为[out_features,in_features]）
    out_features = size(weights.linear1.weight, 1); 
    weights.linear1.bias = zeros(1,out_features);
end

% 加载全连接层2权重并处理偏置
linear2_weight = load('weights\linear2_weight.mat');
weights.linear2.weight = linear2_weight.linear2_weight;
if exist('linear2_bias.mat', 'file')
    linear2_bias = load('linear2_bias.mat');
    weights.linear2.bias = linear2_bias.linear2_bias;
else
    out_features = size(weights.linear2.weight, 1);
    weights.linear2.bias = zeros(1,out_features);
end

% 数据预处理
test_image = double(dataX(:,:,1,1));
[pred_prob, output] = mnist_forward_debug(test_image, weights);
[~, pred_label] = max(pred_prob);
disp(['预测结果：', num2str(pred_label-1)]);

% right = 0;
% for i = 1:500
%     test_image = double(dataX(:,:,1,i)); % 归一化到[0,1]
%     pred_prob = mnist_forward(test_image, weights);
%     real_flag = double(dataY(i));
%     [~, pred_label] = max(pred_prob);
%     if real_flag == pred_label
%         right = right + 1;
%     end
% end


% 前向传播（增加数值稳定性）
function output = mnist_forward(input_image, weights)
    % 用dsODE实现的激活函数
    params = struct;
    params.M = 3;
    params.dt = 1;
    params.V_bin = 1; % Voltage interval size
    params.V_bin_min = -3; % The minimum number of voltage intervals
    params.V_bin_num = params.M /params.V_bin - params.V_bin_min;
    % 卷积层1
    conv1_out = my_conv2d(input_image, weights.conv1.weight, weights.conv1.bias, 1);
    pool1_out = my_maxpool2d(conv1_out, 2);
    relu1_out = dsODE_relu(params, pool1_out);
    
    % 卷积层2
    conv2_out = my_conv2d(relu1_out, weights.conv2.weight, weights.conv2.bias, 1);
    pool2_out = my_maxpool2d(conv2_out, 2);
    relu2_out = dsODE_relu(params, pool2_out);
    
    % 展平
    relu2_permuted = permute(relu2_out, [2,1,3]); % 改变一下顺序
    flattened = reshape(relu2_permuted, [], 1);
    
    % 全连接层（带数值稳定性的Softmax）
    fc1_out = weights.linear1.weight * flattened + weights.linear1.bias';
    %fc1_relu = max(fc1_out, 0);
    
    fc2_out = weights.linear2.weight * fc1_out + weights.linear2.bias';
    % 稳定型Softmax
    max_logit = max(fc2_out);
    stable_logits = fc2_out - max_logit;
    exp_logits = exp(stable_logits);
    output = exp_logits / sum(exp_logits);
end


% 卷积函数（优化维度处理）
function output = my_conv2d(input, weight, bias, stride)
    [H, W, C_in] = size(input);
    [C_out, ~, K, ~] = size(weight);
    out_H = floor((H - K)/stride) + 1;
    out_W = floor((W - K)/stride) + 1;
    
    output = zeros(out_H, out_W, C_out);
    for c_out = 1:C_out
        sum_channel = zeros(out_H, out_W);
        for c_in = 1:C_in
            kernel = squeeze(weight(c_out, c_in, :, :));
            % 有效卷积实现
            for i = 1:out_H
                for j = 1:out_W
                    h_range = (i-1)*stride + (1:K);
                    w_range = (j-1)*stride + (1:K);
                    sum_channel(i,j) = sum_channel(i,j) + ...
                        sum(sum(input(h_range, w_range, c_in) .* kernel));
                end
            end
        end
        output(:,:,c_out) = sum_channel + bias(c_out);
    end
end

% 池化函数（边界检查）
function output = my_maxpool2d(input, pool_size)
    [H, W, C] = size(input);
    out_H = floor(H / pool_size);
    out_W = floor(W / pool_size);
    
    output = zeros(out_H, out_W, C);
    for c = 1:C
        for i = 1:out_H
            for j = 1:out_W
                h_range = (i-1)*pool_size + (1:pool_size);
                w_range = (j-1)*pool_size + (1:pool_size);
                output(i,j,c) = max(max(input(h_range, w_range, c)));
            end
        end
    end
end


function [output, layer_outputs] = mnist_forward_debug(input_image, weights)
    layer_outputs = struct();
    
    % 卷积层1
    conv1_out = my_conv2d(input_image, weights.conv1.weight, weights.conv1.bias, 1);
    layer_outputs.conv1 = conv1_out;
    
    pool1_out = my_maxpool2d(conv1_out, 2);
    layer_outputs.pool1 = pool1_out;
    
    relu1_out = max(pool1_out, 0);
    layer_outputs.relu1 = relu1_out;
    
    % 卷积层2
    conv2_out = my_conv2d(relu1_out, weights.conv2.weight, weights.conv2.bias, 1);
    layer_outputs.conv2 = conv2_out;
    
    pool2_out = my_maxpool2d(conv2_out, 2);
    layer_outputs.pool2 = pool2_out;
    
    relu2_out = max(pool2_out, 0);
    layer_outputs.relu2 = relu2_out;
    
    % 展平
    relu2_permuted = permute(relu2_out, [2,1,3]); % 改变一下顺序
    flattened = reshape(relu2_permuted, [], 1);
    layer_outputs.flatten = flattened;
    
    % 全连接层
    fc1_out = weights.linear1.weight * flattened + weights.linear1.bias';
    layer_outputs.fc1 = fc1_out;
    
    fc2_out = weights.linear2.weight * fc1_out + weights.linear2.bias';
    layer_outputs.fc2 = fc2_out;
    
    % Softmax
    max_logit = max(fc2_out);
    stable_logits = fc2_out - max_logit;
    exp_logits = exp(stable_logits);
    output = exp_logits / sum(exp_logits);
    
    % 保存各层输出
    save('matlab_layer_outputs.mat', '-struct', 'layer_outputs'); 
end