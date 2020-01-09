% NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License Agreement
% Non-Commercial Use Only 
% This NTUITIVE License Agreement, including all exhibits ("NTUITIVE-LA") is a legal agreement between you and NTUITIVE (or “we”) located at 71 Nanyang Drive, NTU Innovation Centre, #01-109, Singapore 637722, a wholly owned subsidiary of Nanyang Technological University (“NTU”) for the software or data identified above, which may include source code, and any associated materials, text or speech files, associated media and "online" or electronic documentation and any updates we provide in our discretion (together, the "Software"). 
% 
% By installing, copying, or otherwise using this Software, found at https://github.com/andriash001/ADL or https://www.researchgate.net/publication/335757711_ADL_Code_mFile, you agree to be bound by the terms of this NTUITIVE-LA.  If you do not agree, do not install copy or use the Software. The Software is protected by copyright and other intellectual property laws and is licensed, not sold.   If you wish to obtain a commercial royalty bearing license to this software please contact us at mpratama@ntu.edu.sg or andriash001@e.ntu.edu.sg.
% 
% SCOPE OF RIGHTS:
% You may use, copy, reproduce, and distribute this Software for any non-commercial purpose, subject to the restrictions in this NTUITIVE-LA. Some purposes which can be non-commercial are teaching, academic research, public demonstrations and personal experimentation. You may also distribute this Software with books or other teaching materials, or publish the Software on websites, that are intended to teach the use of the Software for academic or other non-commercial purposes.
% You may not use or distribute this Software or any derivative works in any form for commercial purposes. Examples of commercial purposes would be running business operations, licensing, leasing, or selling the Software, distributing the Software for use with commercial products, using the Software in the creation or use of commercial products or any other activity which purpose is to procure a commercial gain to you or others.
% If the Software includes source code or data, you may create derivative works of such portions of the Software and distribute the modified Software for non-commercial purposes, as provided herein.  
% If you distribute the Software or any derivative works of the Software, you will distribute them under the same terms and conditions as in this license, and you will not grant other rights to the Software or derivative works that are different from those provided by this NTUITIVE-LA. 
% If you have created derivative works of the Software, and distribute such derivative works, you will cause the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.
% 
% You may not distribute this Software or any derivative works. 
% In return, we simply require that you agree: 
% 1.	That you will not remove any copyright or other notices from the Software.
% 2.	That if any of the Software is in binary format, you will not attempt to modify such portions of the Software, or to reverse engineer or decompile them, except and only to the extent authorized by applicable law. 
% 3.	That NTUITIVE is granted back, without any restrictions or limitations, a non-exclusive, perpetual, irrevocable, royalty-free, assignable and sub-licensable license, to reproduce, publicly perform or display, install, use, modify, post, distribute, make and have made, sell and transfer your modifications to and/or derivative works of the Software source code or data, for any purpose.  
% 4.	That any feedback about the Software provided by you to us is voluntarily given, and NTUITIVE shall be free to use the feedback as it sees fit without obligation or restriction of any kind, even if the feedback is designated by you as confidential. 
% 5.	THAT THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
% 6.	THAT NEITHER NTUITIVE NOR NTU NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS NTUITIVE-LA, INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
% 7.	That we have no duty of reasonable care or lack of negligence, and we are not obligated to (and will not) provide technical support for the Software.
% 8.	That if you breach this NTUITIVE-LA or if you sue anyone over patents that you think may apply to or read on the Software or anyone's use of the Software, this NTUITIVE-LA (and your license and rights obtained herein) terminate automatically.  Upon any such termination, you shall destroy all of your copies of the Software immediately.  Sections 3, 4, 5, 6, 7, 8, 11 and 12 of this NTUITIVE-LA shall survive any termination of this NTUITIVE-LA.
% 9.	That the patent rights, if any, granted to you in this NTUITIVE-LA only apply to the Software, not to any derivative works you make.
% 10.	That the Software may be subject to U.S. export jurisdiction at the time it is licensed to you, and it may be subject to additional export or import laws in other places.  You agree to comply with all such laws and regulations that may apply to the Software after delivery of the software to you.
% 11.	That all rights not expressly granted to you in this NTUITIVE-LA are reserved.
% 12.	That this NTUITIVE-LA shall be construed and controlled by the laws of the Republic of Singapore without regard to conflicts of law.  If any provision of this NTUITIVE-LA shall be deemed unenforceable or contrary to law, the rest of this NTUITIVE-LA shall remain in full effect and interpreted in an enforceable manner that most nearly captures the intent of the original language. 
% 
% Do you accept all of the terms of the preceding NTUITIVE-LA license agreement? If you accept the terms, click “I Agree,” then “Next.”  Otherwise click “Cancel.”
% 
% Copyright (c) NTUITIVE. All rights reserved.

%% list of equation (refer to the paper):
% equation 4.1 is implemented in the line 405
% equation 4.4 is implemented in the line 526
% equation 4.5 is implemented in the line 527
% equation 4.6 is implemented in the line 527
% equation 4.7 is implemented in the line 585
% equation 4.8 is implemented in the line 635
% equation 4.9 is implemented in the line 227 - 242
% equation 4.10 is implemented in the line 238, 239
% equation 4.11 is implemented in the line 244
% equation 4.12 is implemented in the line 305
% equation 4.13 is implemented in the line 181
% equation 4.14 is implemented in the line 415, 418
% equation 4.15 is implemented in the line 419
% equation 4.16 is implemented in the line 416

%% main code
function [parameter,performance] = ADL(data,I,chunkSize,epoch,alpha_w,alpha_d,delta)
%% divide the data into nFolds chunks
dataProportion = 1;     % portion of labeled samples, 0-1
fprintf('=========Autonomous Deep Learning is started=========\n')
[nData,mn] = size(data);
M = mn - I;
l = 0;
nFolds       = round(size(data,1)/chunkSize);                 % number of data chunk
chunk_size   = round(nData/nFolds);
round_nFolds = floor(nData/chunk_size);
Data = {};
if round_nFolds == nFolds
    if nFolds   == 1
        Data{1} = data;
    else
        for i=1:nFolds
            l = l+1;
            if i < nFolds
                Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            elseif i == nFolds
                Data1 = data(((i-1)*chunk_size+1):end,:);
            end
            Data{l} = Data1;
        end
    end
else
    if nFolds == 1
        Data{1} = data;
    else
        for i=1:nFolds-1
            l=l+1;
            Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            Data{l} = Data1;
        end
        i = i + 1;
        Data{nFolds} = data(((i-1)*chunk_size+1):end,:);
    end
end
buffer_x = [];buffer_T = [];tTest = []; acttualLabel = []; classPerdiction = [];
clear data Data1

%% initiate model
K = 1;          %initial node
parameter.net = netInit([I K M]);

%% initiate node evolving iterative parameters
layer                       = 1;     % number of layer
parameter.ev{1}.layer       = layer;
parameter.ev{1}.kp          = 0;
parameter.ev{1}.miu_x_old   = 0;
parameter.ev{1}.var_x_old   = 0;
parameter.ev{1}.kl          = 0;
parameter.ev{1}.K           = K;
parameter.ev{1}.cr          = 0;
parameter.ev{1}.node        = [];
parameter.ev{1}.BIAS2       = [];
parameter.ev{1}.VAR         = [];
parameter.ev{1}.miu_NS_old  = 0;
parameter.ev{1}.var_NS_old  = 0;
parameter.ev{1}.miu_NHS_old = 0;
parameter.ev{1}.var_NHS_old = 0;
parameter.ev{1}.miumin_NS   = [];
parameter.ev{1}.miumin_NHS  = [];
parameter.ev{1}.stdmin_NS   = [];
parameter.ev{1}.stdmin_NHS  = [];

%% initiate drift detection parameter
alpha = alpha_d;

%% initiate layer merging iterative parameters
for k3=1:M
    covariance(1,:,k3) = 0;
    covariance(:,1,k3) = 0;
end
covariance_old             = covariance;
threshold                  = delta;      % similarity measure
parameter.prune_list       = 0;
parameter.prune_list_index = [];

%% main loop, prequential evaluation
for iFolds = 1:nFolds
    %% load the data chunk-by-chunk
    x = Data{iFolds}(:,1:I);
    T = Data{iFolds}(:,I+1:mn);
    [bd,~] = size(T);
    clear Data{t}
    
    %% neural network testing
    start_test = tic;
    fprintf('=========Chunk %d of %d=========\n', iFolds, size(Data,2))
    disp('Discriminative Testing: running ...');
    parameter.net.t = iFolds;
    [parameter.net] = testing(parameter.net,x,T,parameter.ev);
    parameter.net.test_time(iFolds) = toc(start_test);
    
    % metrics calculation
    parameter.Loss(iFolds) = parameter.net.loss(parameter.net.index);
    tTest(bd*iFolds+(1-bd):bd*iFolds,:) = parameter.net.sigma;
    % if iFolds > 1
    %     acttualLabel    = [acttualLabel parameter.net.acttualLabel'];
    %     classPerdiction = [classPerdiction parameter.net.classPerdiction'];
    % end
    parameter.residual_error(bd*iFolds+(1-bd):bd*iFolds,:) = parameter.net.residual_error;
    parameter.cr(iFolds) = parameter.net.cr;
    
    % performance measure
    [performance.ev.f_measure(iFolds,:),performance.ev.g_mean(iFolds,:),performance.ev.recall(iFolds,:),performance.ev.precision(iFolds,:),performance.ev.err(iFolds,:)] = performanceMeasure(parameter.net.acttualLabel, parameter.net.classPerdiction, M);
    if iFolds == nFolds
        fprintf('=========Parallel Autonomous Deep Learning is finished=========\n')
        break               % last chunk only testing
    end
    
    %% Layer pruning mechanism
    start_train = tic;
    outputCovar = zeros(layer,layer,M);
    for iter = 1:layer
        for iter1 = 1:layer
            if parameter.net.beta(iter) ~= 0 && parameter.net.beta(iter1) ~= 0
                for iter2 = 1:M
                    temporary = cov(parameter.net.activityOutput{iter1}(:,iter2),parameter.net.activityOutput{iter}(:,iter2));
                    outputCovar(iter,iter1,iter2) = temporary(1,2);
                    covariance (iter,iter1,iter2) = (covariance_old(iter,iter1,iter2)*(iFolds - 1) + (((iFolds - 1)/iFolds)*outputCovar(iter,iter1,iter2)))/iFolds;
                end
            end
        end
    end
    covariance_old = covariance;
    if layer > 1
        merged_list = [];
        for l = 0:layer - 2
            for hh = 1:layer - l - 1
                if parameter.net.beta(end - l) ~= 0 || parameter.net.beta(hh) ~= 0        % only for parallel
                    MCI = zeros(1,M);
                    for o = 1:M
                        pearson = covariance(end - l,hh,o)/sqrt(covariance(end - l,end - l,o)*covariance(hh,hh,o));
                        MCI(o)  = (0.5*(covariance(hh,hh,o) + covariance(end - l,end - l,o)) - sqrt((covariance(hh,hh,o) + covariance(end - l,end - l,o))^(2) - 4*covariance(end - l,end - l,o)*covariance(hh,hh,o)*(1 - pearson^(2))));
                    end
                    if max(abs(MCI)) < threshold    % equation 4.13
                        if isempty(merged_list)
                            merged_list(1,1) = layer - l;
                            merged_list(1,2) = hh;
                        else
                            No  = find(merged_list(:,1:end - 1) == layer - l, 1);
                            No1 = find(merged_list(:,1:end - 1) == hh, 1);
                            if isempty(No) && isempty(No1)
                                merged_list(end + 1,1) = layer - l;
                                merged_list(end + 1,2) = hh;
                            end
                        end
                        break
                    end
                end
            end
        end
        del_list = [];
        for i = 1:size(merged_list,1)
            noOfHighlyCorrelated = find(merged_list(i,:) == 0, 1);
            if isempty(noOfHighlyCorrelated)
                if parameter.net.beta(merged_list(i,1)) > parameter.net.beta(merged_list(i,2))
                    deleteLayer = merged_list(i,2);
                else
                    deleteLayer = merged_list(i,1);
                end
                del_list = [del_list deleteLayer];
            end
        end
        if isempty(del_list) == false && parameter.net.beta(del_list) ~= 0
            fprintf('The Hidden Layer no %d is PRUNED around chunk %d\n', del_list, iFolds)
            parameter.net.beta(del_list) = 0;
        end
        parameter.prune_list       = parameter.prune_list + length(del_list);
        parameter.prune_list_index = [parameter.prune_list_index del_list];
    end
    
    %% Drift detection: output space
    if iFolds > 1
        cuttingpoint = 0;
        pp    = length(T);                      % batch size
        F_cut = zeros(pp,1);                    % initiate accuracy matrix F
        F_cut(parameter.net.wrongClass,:) = 1;
        [Fupper,~] = max(F_cut);
        [Flower,~] = min(F_cut);
        miu_F = mean(F_cut);
        for cut = 1:pp
            % finding the cutting point, following equation 4.9
            miu_G = mean(F_cut(1:cut,:));
            [Gupper,~] = max(F_cut(1:cut,:));
            [Glower,~] = min(F_cut(1:cut,:));
            epsilon_G  = (Gupper - Glower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha)));
            epsilon_F  = (Fupper - Flower)*sqrt(((pp)/(2*cut*(pp))*log(1/alpha)));
            if (epsilon_G + miu_G) >= (miu_F + epsilon_F)
                cuttingpoint = cut;
                miu_H = mean(F_cut(cuttingpoint+1:end,:));
                % calculate the epsilon W and epsilon D using equation 4.10
                epsilon_D = (Fupper-Flower)*sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)))*log(1/alpha_d));
                epsilon_W = (Fupper-Flower)*sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)))*log(1/alpha_w));
                break
            end
        end
        if cuttingpoint == 0
            miu_H = 0;
            epsilon_D = 0;
            epsilon_W = 0;
        end
        if abs(miu_G - miu_H) > epsilon_D && cuttingpoint > 1 && cuttingpoint < pp
            % abs(miu_G - miu_H) > epsilon_D is the equation 4.11
            st = 1;
            disp('Drift state: DRIFT');
            layer                      = layer + 1;
            parameter.net.nLayer       = parameter.net.nLayer + 1;
            parameter.net.nHiddenLayer = layer;
            parameter.net.index        = parameter.net.nHiddenLayer;
            fprintf('The new Layer no %d is FORMED around chunk %d\n', layer, iFolds)
            
            % initiate NN weight parameters
            [ii,~] = size(parameter.net.weight{layer-1});
            parameter.net.weight {layer}  = normrnd(0,sqrt(2/(ii+1)),[1,ii+1]);
            parameter.net.velocity{layer} = zeros(1,ii+1);
            parameter.net.grad{layer}     = zeros(1,ii+1);
            
            % initiate new classifier weight
            parameter.net.weightSoftmax {layer}  = normrnd(0,1,[M,2]);
            parameter.net.momentumSoftmax{layer} = zeros(M,2);
            parameter.net.gradSoftmax{layer}     = zeros(M,2);
            
            % initiate new voting weight
            parameter.net.beta(layer)    = 1;
            parameter.net.betaOld(layer) = 1;
            parameter.net.p(layer)       = 1;
            
            % initiate iterative parameters
            parameter.ev{layer}.layer       = layer;
            parameter.ev{layer}.kl          = 0;
            parameter.ev{layer}.K           = 1;
            parameter.ev{layer}.cr          = 0;
            parameter.ev{layer}.node        = [];
            parameter.ev{layer}.miu_NS_old  = 0;
            parameter.ev{layer}.var_NS_old  = 0;
            parameter.ev{layer}.miu_NHS_old = 0;
            parameter.ev{layer}.var_NHS_old = 0;
            parameter.ev{layer}.miumin_NS   = [];
            parameter.ev{layer}.miumin_NHS  = [];
            parameter.ev{layer}.stdmin_NS   = [];
            parameter.ev{layer}.stdmin_NHS  = [];
            parameter.ev{layer}.BIAS2       = [];
            parameter.ev{layer}.VAR         = [];
            
            % initiate covariance for rule merging
            for k3=1:M        
                covariance(layer,:,k3) = 0;
                covariance(:,layer,k3) = 0;
            end
            covariance_old = covariance;
            
            if isempty(buffer_x)    % check buffer
            else
                % prepare input for training
                x = [parameter.net.activity{1}(:,2:end);buffer_x]; 
                T = [T;buffer_T];
                parameter.net.T = T;
                parameter.net = netFeedForward(parameter.net, x, T);
            end
            
            buffer_x = []; % clear the buffer of input
            buffer_T = []; % clear the buffer of output
        elseif abs(miu_G - miu_H) >= epsilon_W && abs(miu_G - miu_H) < epsilon_D && st ~= 2
            % abs(miu_G - miu_H) >= epsilon_W && abs(miu_G - miu_H) < epsilon_D is the equation 4.12
            disp('Drift state: WARNING');
            st       = 2;
            
            buffer_x = x; % save current data batch to buffer of input
            buffer_T = T; % save current data batch to buffer of output
        else
            st = 3;
            disp('Drift state: STABLE');
            
            if isempty(buffer_x)    % check buffer
            else
                x  = [buffer_x;x];  % add data buffer to training data
                T  = [buffer_T;T];  % add data buffer to training data
                parameter.net.T = T;
                parameter.net   = netFeedForward(parameter.net, x, T);
            end
            
            buffer_x = []; % clear the buffer of input
            buffer_T = []; % clear the buffer of output
        end
    else
        st = 3;
        disp('Drift state: STABLE');
        buffer_x = [];
        buffer_T = [];
    end
    driftState(iFolds) = st;
    nHidLayer(iFolds) = numel(find(parameter.net.beta ~= 0));
    parameter.wl(iFolds) = parameter.net.index;
    
    %% training for winning layer
    if st ~= 2
        parameter = training(parameter,T,epoch,dataProportion);
        fprintf('=========Hidden layer number %d was updated=========\n', parameter.net.index)
    end
    parameter.net.update_time(iFolds) = toc(start_train);
    
    %% clear current chunk data
    clear Data{t}
    parameter.net.activity = {};
    
end

%% statistical measure
% [performance.f_measure,performance.g_mean,performance.recall,performance.precision,performance.err] = performanceMeasure(acttualLabel, classPerdiction, M);

%% save the numerical result
parameter.drift         = driftState;
parameter.nFolds        = nFolds;
performance.update_time = [mean(parameter.net.update_time) std(parameter.net.update_time)];
performance.test_time   = [mean(parameter.net.test_time) std(parameter.net.test_time)];
performance.classification_rate = [mean(parameter.cr(2:end)) std(parameter.cr(2:end))];
performance.layer               = [mean(nHidLayer) std(nHidLayer)];
performance.LayerWeight         = parameter.net.beta;
meanode                         = [];
stdnode                         = [];
for i = 1:parameter.net.nHiddenLayer
    a = nnz(~parameter.net.nodes{i});
    parameter.net.nodes{i} = parameter.net.nodes{i}(a+1:iFolds);
    meanode = [meanode mean(parameter.net.nodes{i})];
    stdnode = [stdnode std(parameter.net.nodes{i})];
end
performance.meanode = meanode;
performance.stdnode = stdnode;
performance.NumberOfParameters = parameter.net.mnop;
parameter.HL = nHidLayer;

%% plot the result
subplot(3,1,1)
plot(parameter.cr)
ylim([0 1.1]);
xlim([1 nFolds]);
ylabel('Classification Rate')
subplot(3,1,2)
plot(parameter.Loss)
ylim([0 1.1]);
xlim([1 nFolds]);
ylabel('Loss')
subplot(3,1,3)
plot(nHidLayer)
ylabel('No of hidden layer')
xlim([1 nFolds]);
xlabel('chunk');
hold off
fprintf('=========Autonomous Deep Learning is finished=========\n')
end

%% testing phase of ADL
% This function is developed with modification from Rasmus Berg Palm
% Copyright (c) 2012, Rasmus Berg Palm (rasmusbergpalm@gmail.com)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% 
% Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
% OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
% IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
function [net] = testing(net, input, trueClass, ev)
%% feedforward
net              = netFeedForward(net, input, trueClass);
[nData,m2]       = size(trueClass);
decreasingFactor = 0.001;

%% obtain acttual Label
[~,acttualLabel] = max(trueClass,[],2);
net.sigma = zeros(nData,m2);
for iData = 1 : nData
    for iHiddenLayer = 1 : net.nHiddenLayer
        if net.beta(iHiddenLayer) ~= 0
            %% obtain the predicted label, according equation 4.1
            % note that the layer weight betaOld is fixed, obtained from
            % the previous batch
            net.sigma(iData,:) = net.sigma(iData,:) + net.activityOutput{iHiddenLayer}(iData,:)*net.betaOld(iHiddenLayer);
            [~, net.classlabel{iHiddenLayer}(iData,:)] = max(net.activityOutput{iHiddenLayer}(iData,:),[],2);
            
            %% train the dynamic voting weight beta
            compare = acttualLabel(iData,:) - net.classlabel{iHiddenLayer}(iData,:);
            if compare ~= 0
                net.p(iHiddenLayer)    = max(net.p(iHiddenLayer)-decreasingFactor,decreasingFactor);        % wrong prediction decrease p, equation 4.14
                net.beta(iHiddenLayer) = max(net.beta(iHiddenLayer)*net.p(iHiddenLayer),decreasingFactor);  % penalty, equation 4.16
            elseif compare == 0
                net.p(iHiddenLayer)    = min(net.p(iHiddenLayer)+decreasingFactor,1);                       % correct prediction increase p, equation 4.14
                net.beta(iHiddenLayer) = min(net.beta(iHiddenLayer)*(1+net.p(iHiddenLayer)),1);             % reward, equation 4.15
            end
        end
        
        if iData == nData
            %% calculate the number of parameter
            if net.beta(iHiddenLayer) ~= 0
                [c,d] = size(net.weightSoftmax{iHiddenLayer});
                vw = 1;
            else
                c = 0;
                d = 0;
                vw = 0;
            end
            [a,b] = size(net.weight{iHiddenLayer});
            nop(iHiddenLayer) = a*b + c*d + vw;     % no of parameters
            
            %% calculate the number of node in each hidden layer
            net.nodes{iHiddenLayer}(net.t) = ev{iHiddenLayer}.K;
        end
    end
end
net.nop(net.t) = sum(nop);
net.mnop       = [mean(net.nop) std(net.nop)];

%% update the voting weight
net.beta      = net.beta/sum(net.beta);
net.betaOld   = net.beta;
[~,net.index] = max(net.beta);

%% calculate classification rate
[multiClassProb,classPerdiction] = max(net.sigma,[],2);
net.wrongClass      = find(classPerdiction ~= acttualLabel);
net.cr              = 1 - numel(net.wrongClass)/nData;
net.residual_error  = 1 - multiClassProb;
net.classPerdiction = classPerdiction;
net.acttualLabel    = acttualLabel;
end

%% train the winning layer ADL
function parameter  = training(parameter,y,nEpoch,dataProportion)
[~,bb] = size(parameter.net.weight{parameter.net.index});
grow   = 0;
prune  = 0;

%% initiate performance matrix
ly          = parameter.net.index;
kp          = parameter.ev{1}.kp;
miu_x_old   = parameter.ev{1}.miu_x_old;
var_x_old   = parameter.ev{1}.var_x_old;
kl          = parameter.ev{ly}.kl;
K           = parameter.ev{ly}.K;
node        = parameter.ev{ly}.node;
growingThreshold = parameter.ev{ly}.BIAS2;
pruningThreshold = parameter.ev{ly}.VAR;
miu_NS_old  = parameter.ev{ly}.miu_NS_old;
var_NS_old  = parameter.ev{ly}.var_NS_old;
miu_NHS_old = parameter.ev{ly}.miu_NHS_old;
var_NHS_old = parameter.ev{ly}.var_NHS_old;
miumin_NS   = parameter.ev{ly}.miumin_NS;
miumin_NHS  = parameter.ev{ly}.miumin_NHS;
stdmin_NS   = parameter.ev{ly}.stdmin_NS;
stdmin_NHS  = parameter.ev{ly}.stdmin_NHS;

%% initiate training model
net                    = netInitWinner([1 1 1]);
net.activationFunction = parameter.net.activationFunction;
net.output             = parameter.net.output;

%% substitute the weight to be trained to training model
net.weight{1}   = parameter.net.weight{ly};
net.velocity{1} = parameter.net.velocity{ly};
net.grad{1}     = parameter.net.grad{ly};
net.weight{2}   = parameter.net.weightSoftmax{ly};
net.velocity{2} = parameter.net.momentumSoftmax{ly};
net.grad{2}     = parameter.net.gradSoftmax{ly};

%% load the data for training
x = parameter.net.activity{ly};
[nData,I]   = size(x);
kk          = randperm(nData);
x           = x(kk,:);
y           = y(kk,:);
nLabeledData= round(dataProportion*nData);
x           = x(1:nLabeledData,:);
y           = y(1:nLabeledData,:);
[nData,~]   = size(x);


%% xavier initialization constant
if ly > 1
    n_in = parameter.ev{ly-1}.K;
else
    n_in = parameter.net.initialConfig(1);
end

%% main loop, train the model
for iData = 1 : nData
    kp = kp + 1;
    kl = kl + 1;
    
    %% Incremental calculation of bias and variance
    [miu_x,std_x,var_x] = meanstditer(miu_x_old,var_x_old,parameter.net.activity{1}(iData,:),kp);
    miu_x_old = miu_x;
    var_x_old = var_x;
    
    % Expectation of output
    py = probit(miu_x,std_x)';        % This implements equation 4.4
    for ii = 1:parameter.net.index
        if ii == parameter.net.index
            py = sigmf(net.weight{1}*py,[1,0]);
        else
            py = sigmf(parameter.net.weight{ii}*py,[1,0]);
        end
        py = [1;py];
        if ii == 1
            Ey2 = py.^2;
        end
    end % this implements equation 4.5, 4.6
    Ey = py;
    Ez = net.weight{2}*Ey;
    Ez = exp(Ez - max(Ez));
    Ez = Ez./sum(Ez);           % E[output]
    
    % expectation of output2
    if parameter.net.nHiddenLayer > 1
        py = Ey2;
        for ii = 2:parameter.net.index
            if ii == parameter.net.index
                py = sigmf(net.weight{1}*py,[1,0]);
            else
                py = sigmf(parameter.net.weight{ii}*py,[1,0]);
            end
            py = [1;py];
        end
        Ey2 = py;
    end
    Ez2 = net.weight{2}*Ey2;
    Ez2 = exp(Ez2 - max(Ez2));
    Ez2 = Ez2./sum(Ez2);        % E[output2]
    
    %% Network mean calculation
    bias2 = (Ez - y(iData,:)').^2;  % bias = (E[y] - y)2
    ns    = bias2;
    NS    = norm(ns,'fro');         % norm operator to summarize
    
    %% Incremental calculation of NS mean and variance
    [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kl);
    miu_NS_old = miu_NS;            % empirical mean of bias
    var_NS_old = var_NS;
    miustd_NS  = miu_NS + std_NS;
    if kl <= 1 || grow == 1
        miumin_NS = miu_NS;
        stdmin_NS = std_NS;
    else
        if miu_NS < miumin_NS
            miumin_NS = miu_NS;
        end
        if std_NS < stdmin_NS
            stdmin_NS = std_NS;
        end
    end
    miustdmin_NS  = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS; % right hand side of equation 4.7
    growingThreshold(kp,:)   = miu_NS;     
    
    %% growing hidden unit if equation 4.7 is satisfied
    if miustd_NS >= miustdmin_NS && kl > 1
        grow            = 1;
        K               = K + 1;
        fprintf('The new node no %d is FORMED around sample %d\n', K, kp)
        node(kp)        = K;
        
        % augment the weight
        net.weight{1}   = [net.weight{1};normrnd(0,sqrt(2/(n_in+1)),[1,bb])];
        net.velocity{1} = [net.velocity{1};zeros(1,bb)];
        net.grad{1}     = [net.grad{1};zeros(1,bb)];
        net.weight{2}   = [net.weight{2} normrnd(0,sqrt(2/(K+1)),[parameter.net.initialConfig(end),1])];
        net.velocity{2} = [net.velocity{2} zeros(parameter.net.initialConfig(end),1)];
        net.grad{2}     = [net.grad{2} zeros(parameter.net.initialConfig(end),1)];
        if ly < parameter.net.nHiddenLayer
            % add weight to the following hidden layer, if the winning
            % layer is not the last hidden layer
            [wNext,~]                    = size(parameter.net.weight{ly+1});
            parameter.net.weight{ly+1}   = [parameter.net.weight{ly+1} normrnd(0,sqrt(2/(K+1)),[wNext,1])];
            parameter.net.velocity{ly+1} = [parameter.net.velocity{ly+1} zeros(wNext,1)];
            parameter.net.grad{ly+1}     = [parameter.net.grad{ly+1} zeros(wNext,1)];
        end
    else
        grow     = 0;
        node(kp) = K;
    end
    
    %% Network variance calculation
    var = Ez2 - Ez.^2;          % variance
    NHS = norm(var,'fro');
    
    %% Incremental calculation of NHS mean and variance
    [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kl);
    miu_NHS_old = miu_NHS;      % empirical mean of variance
    var_NHS_old = var_NHS;
    miustd_NHS  = miu_NHS + std_NHS;
    if kl <= I+1 || prune == 1
        miumin_NHS = miu_NHS;
        stdmin_NHS = std_NHS;
    else
        if miu_NHS < miumin_NHS
            miumin_NHS = miu_NHS;
        end
        if std_NHS < stdmin_NHS
            stdmin_NHS = std_NHS;
        end
    end
    miustdmin_NHS  = miumin_NHS + (2.6*exp(-NHS)+1.4)*stdmin_NHS;  % right hand side of equation 4.8
    pruningThreshold(kp,:)      = miu_NHS;
    
    %% pruning hidden unit, if equation 4.8 is satisfied
    if grow == 0 && K > 1 && miustd_NHS >= miustdmin_NHS && kl > I + 1
        HS       = Ey(2:end);
        [~,BB]   = min(HS);     % find the least contributing hidden unit
        fprintf('The node no %d is PRUNED around sample %d\n', BB, kp)
        prune    = 1;
        K        = K - 1;
        node(kp) = K;
        
        % delete the weight
        net.weight{1}(BB,:)   = [];
        net.velocity{1}(BB,:) = [];
        net.grad{1}(BB,:)     = [];
        net.weight{2}(:,BB+1)   = [];
        net.velocity{2}(:,BB+1) = [];
        net.grad{2}(:,BB+1)     = [];
        if ly < parameter.net.nHiddenLayer
            % remove weight to the following hidden layer, if the winning
            % layer is not the last hidden layer
            parameter.net.weight{ly+1}(:,BB+1)   = [];
            parameter.net.velocity{ly+1}(:,BB+1) = [];
            parameter.net.grad{ly+1}(:,BB+1)     = [];
        end
    else
        node(kp) = K;
        prune = 0;
    end
    
    %% feedforward
    net = netFeedForwardWinner(net, x(iData,:), y(iData,:));
    
    %% optimize the parameters
    net = lossBackward(net);
    net = optimizerStep(net);
end

%% iterative learning
if nEpoch > 1
    for iEpoch = 1:nEpoch-1
        kk = randperm(nData);
        x = x(kk,:);
        y = y(kk,:);
        for iData = 1 : nData
            %% feedforward
            net = netFeedForwardWinner(net, x(iData,:), y(iData,:));
            
            %% optimize the parameters
            net = lossBackward(net);
            net = optimizerStep(net);
        end
    end
end

%% substitute the weight back to main model
parameter.net.weight{ly}         = net.weight{1};
parameter.net.weightSoftmax{ly}  = net.weight{2};

%% reset momentumCoeff and gradient
parameter.net.velocity{ly}        = net.velocity{1}*0;
parameter.net.grad{ly}            = net.grad{1}*0;
parameter.net.momentumSoftmax{ly} = net.velocity{2}*0;
parameter.net.gradSoftmax{ly}     = net.grad{2}*0;

%% substitute the recursive calculation
parameter.ev{1}.kp           = kp;
parameter.ev{1}.miu_x_old    = miu_x_old;
parameter.ev{1}.var_x_old    = var_x_old;
parameter.ev{ly}.kl          = kl;
parameter.ev{ly}.K           = K;
parameter.ev{ly}.node        = node;
parameter.ev{ly}.BIAS2       = growingThreshold;
parameter.ev{ly}.VAR         = pruningThreshold;
parameter.ev{ly}.miu_NS_old  = miu_NS_old;
parameter.ev{ly}.var_NS_old  = var_NS_old;
parameter.ev{ly}.miu_NHS_old = miu_NHS_old;
parameter.ev{ly}.var_NHS_old = var_NHS_old;
parameter.ev{ly}.miumin_NS   = miumin_NS;
parameter.ev{ly}.miumin_NHS  = miumin_NHS;
parameter.ev{ly}.stdmin_NS   = stdmin_NS;
parameter.ev{ly}.stdmin_NHS  = stdmin_NHS;
end

%% feedforward operation
% This function is developed with modification from Rasmus Berg Palm
% Copyright (c) 2012, Rasmus Berg Palm (rasmusbergpalm@gmail.com)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% 
% Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
% OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
% IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
function net = netFeedForward(net, x, output)
nLayer    = net.nLayer;
batchSize = size(x,1);
x         = [ones(batchSize,1) x];  % by adding 1 to the first coulomn, it means the first coulomn of weight is bias
net.activity{1} = x;                % the first activity is the input itself

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activationFunction
        case 'sigmf'
            net.activity{iLayer} = sigmf(net.activity{iLayer - 1} * net.weight{iLayer - 1}',[1,0]);
        case 'tanh'
            net.activity{iLayer} = tanh(net.activity{iLayer - 1} * net.weight{iLayer - 1}');
        case 'relu'
            net.activity{iLayer} = max(net.activity{iLayer - 1} * net.weight{iLayer - 1}',0);
    end
    net.activity{iLayer} = [ones(batchSize,1) net.activity{iLayer}];
end

%% propagate to the output layer
for iLayer = 1 : net.nHiddenLayer
    if net.beta(iLayer) ~= 0
        switch net.output
            case 'sigmf'
                net.activityOutput{iLayer} = sigmf(net.activity{iLayer + 1} * net.weightSoftmax{iLayer}',[1,0]);
            case 'linear'
                net.activityOutput{iLayer} = net.activity{iLayer + 1} * net.weightSoftmax{iLayer}';
            case 'softmax'
                net.activityOutput{iLayer} = stableSoftmax(net.activity{iLayer + 1},net.weightSoftmax{iLayer});
        end
        
        %% calculate error
        net.error{iLayer} = output - net.activityOutput{iLayer};
        
        %% calculate loss function
        switch net.output
            case {'sigmf', 'linear'}
                net.loss(iLayer) = 1/2 * sum(sum(net.error{iLayer} .^ 2)) / batchSize;
            case 'softmax'
                net.loss(iLayer) = -sum(sum(output .* log(net.activityOutput{iLayer}))) / batchSize;
        end
    end
end
end

%% initiate neural network
% This function is developed with modification from Rasmus Berg Palm
% Copyright (c) 2012, Rasmus Berg Palm (rasmusbergpalm@gmail.com)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% 
% Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
% OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
% IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
function net = netInit(layer)
net.initialConfig        = layer;
net.nLayer               = numel(net.initialConfig);  %  Number of layer
net.nHiddenLayer         = net.nLayer - 2;   %  number of hidden layer
net.activationFunction   = 'sigmf';          %  Activation functions of hidden layers: 'sigmoid function', 'tanh' and 'relu'.
net.learningRate         = 0.01;             %  learning rate smaller value is preferred
net.momentumCoeff        = 0.95;             %  Momentum coefficient, higher value is preferred
net.outputConnect        = 1;                %  1: connect all hidden layer output to output layer, otherwise: only the last hidden layer is connected to output
net.output               = 'softmax';        %  output layer can be selected as follows: 'sigmf', 'softmax', and 'linear'

%% initiate weights and weight momentumCoeff for hidden layer
for iLayer = 2 : net.nLayer - 1
    net.weight {iLayer - 1}  = normrnd(0,sqrt(2/(net.initialConfig(iLayer-1)+1)),[net.initialConfig(iLayer),net.initialConfig(iLayer - 1)+1]);
    net.velocity{iLayer - 1} = zeros(size(net.weight{iLayer - 1}));
    net.grad{iLayer - 1}     = zeros(size(net.weight{iLayer - 1}));
    net.c{iLayer - 1}        = normrnd(0,sqrt(2/(net.initialConfig(iLayer-1)+1)),[net.initialConfig(iLayer - 1),1]);
end

%% initiate weights and weight momentumCoeff for output layer
for iHiddenLayer = 1 : net.nHiddenLayer
    net.weightSoftmax {iHiddenLayer}   = normrnd(0,sqrt(2/(size(net.weight{iHiddenLayer},1)+1)),[net.initialConfig(end),net.initialConfig(iHiddenLayer+1)+1]);
    net.momentumSoftmax{iHiddenLayer}  = zeros(size(net.weightSoftmax{iHiddenLayer}));
    net.gradSoftmax{iHiddenLayer}      = zeros(size(net.weightSoftmax{iHiddenLayer}));
    net.beta(iHiddenLayer)             = 1;
    net.betaOld(iHiddenLayer)          = 1;
    net.p(iHiddenLayer)                = 1;
end
end

%% calculate recursive mean and standard deviation
function [miu,std,var] = meanstditer(miu_old,var_old,x,k)
miu = miu_old + (x - miu_old)./k;
var = var_old + (x - miu_old).*(x - miu);
std = sqrt(var/k);
end

%% calculate probit function
function p = probit(miu,std)
p = (miu./(1 + pi.*(std.^2)./8).^0.5);
end

%% stable softmax
function output = stableSoftmax(activation,weight)
output = activation * weight';
output = exp(output - max(output,[],2));
output = output./sum(output, 2);
end

function net = netInitWinner(layer)
net.initialConfig   = layer;                    %  winning layer
net.nLayer          = numel(net.initialConfig); %  Number of layer
net.learningRate    = 0.01;                     %  learning rate, smaller value is preferred
net.momentumCoeff   = 0.95;                     %  Momentum coefficient, higher value is preferred
end

%% feedforward for a single layer network. It is used to train the winning layer
% This function is developed with modification from Rasmus Berg Palm
% Copyright (c) 2012, Rasmus Berg Palm (rasmusbergpalm@gmail.com)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% 
% Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
% OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
% IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
function net = netFeedForwardWinner(net, input, output)
nLayer = net.nLayer;
batchSize = size(input,1);
net.activity{1} = input;

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activationFunction
        case 'sigmf'
            net.activity{iLayer} = sigmf(net.activity{iLayer - 1} * net.weight{iLayer - 1}',[1,0]);
        case 'tanh'
            net.activity{iLayer} = tanh(net.activity{iLayer - 1} * net.weight{iLayer - 1}');
        case 'relu'
            net.activity{iLayer} = max(net.activity{iLayer - 1} * net.weight{iLayer - 1}',0);
    end
    net.activity{iLayer}         = [ones(batchSize,1) net.activity{iLayer}];  % augment with ones, act as bias multiplier
end

%% propagate to the output layer
switch net.output
    case 'sigmf'
        net.activity{nLayer} = sigmf(net.activity{nLayer - 1} * net.weight{nLayer - 1}',[1,0]);
    case 'linear'
        net.activity{nLayer} = net.activity{nLayer - 1} * net.weight{nLayer - 1}';
    case 'softmax'
        net.activity{nLayer} = stableSoftmax(net.activity{nLayer - 1},net.weight{nLayer - 1});
end

%% calculate error
net.error = output - net.activity{nLayer};
end

%% calculate backpropagation of the network
% This function is developed with modification from Rasmus Berg Palm
% Copyright (c) 2012, Rasmus Berg Palm (rasmusbergpalm@gmail.com)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% 
% Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
% OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
% IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
function net = lossBackward(net)
nLayer = net.nLayer;

%% error backward
switch net.output
    case 'sigmf'
        backPropSignal{nLayer} = - net.error .* (net.activity{nLayer} .* (1 - net.activity{nLayer}));
    case {'softmax','linear'}
        backPropSignal{nLayer} = - net.error;          % loss derivative w.r.t. output
end

%% activation backward
for iLayer = (nLayer - 1) : -1 : 2
    switch net.activationFunction
        case 'sigmf'
            actFuncDerivative = net.activity{iLayer} .* (1 - net.activity{iLayer}); % contains b
        case 'tanh'
            actFuncDerivative = 1 - net.activity{iLayer}.^2;
        case 'relu'
            actFuncDerivative = zeros(1,length(net.activity{iLayer}));
            actFuncDerivative(net.activity{iLayer}>0) = 0.1;
    end
    
    if iLayer+1 == nLayer
        backPropSignal{iLayer} = (backPropSignal{iLayer + 1} * net.weight{iLayer}) .* actFuncDerivative;
    else
        backPropSignal{iLayer} = (backPropSignal{iLayer + 1}(:,2:end) * net.weight{iLayer}) .* actFuncDerivative;
    end
end

%% calculate gradient
for iLayer = 1 : (nLayer - 1)
    if iLayer + 1 == nLayer
        net.grad{iLayer} = (backPropSignal{iLayer + 1}' * net.activity{iLayer});
    else
        net.grad{iLayer} = (backPropSignal{iLayer + 1}(:,2:end)' * net.activity{iLayer});
    end
end
end

%% update the weight
% This function is developed with modification from Rasmus Berg Palm
% Copyright (c) 2012, Rasmus Berg Palm (rasmusbergpalm@gmail.com)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% 
% Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
% OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
% IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
function net = optimizerStep(net)
for iLayer = 1 : (net.nLayer - 1)
    grad = net.grad{iLayer};
    net.velocity{iLayer} = net.momentumCoeff*net.velocity{iLayer} + net.learningRate * grad;
    finalGrad            = net.velocity{iLayer};
    
    %% apply the gradient to the weight
    net.weight{iLayer} = net.weight{iLayer} - finalGrad;
end
end

%% Performance measure
% This function is developed from Gregory Ditzler
% https://github.com/gditzler/IncrementalLearning/blob/master/src/stats.m
function [fMeasure,gMean,recall,precision,error] = performanceMeasure(trueClass, rawOutput, nClass)
label           = index2vector(trueClass, nClass);
predictedLabel  = index2vector(rawOutput, nClass);

recall      = calculate_recall(label, predictedLabel, nClass);
error       = 1 - sum(diag(predictedLabel'*label))/sum(sum(predictedLabel'*label));
precision   = calculate_precision(label, predictedLabel, nClass);
gMean       = calculate_g_mean(recall, nClass);
fMeasure    = calculate_f_measure(label, predictedLabel, nClass);


    function gMean = calculate_g_mean(recall, nClass)
        gMean = (prod(recall))^(1/nClass);
    end

    function fMeasure = calculate_f_measure(label, predictedLabel, nClass)
        fMeasure = zeros(1, nClass);
        for iClass = 1:nClass
            fMeasure(iClass) = 2*label(:, iClass)'*predictedLabel(:, iClass)/(sum(predictedLabel(:, iClass)) + sum(label(:, iClass)));
        end
        fMeasure(isnan(fMeasure)) = 1;
    end

    function precision = calculate_precision(label, predictedLabel, nClass)
        precision = zeros(1, nClass);
        for iClass = 1:nClass
            precision(iClass) = label(:, iClass)'*predictedLabel(:, iClass)/sum(predictedLabel(:, iClass));
        end
        precision(isnan(precision)) = 1;
    end

    function recall = calculate_recall(label, predictedLabel, nClass)
        recall = zeros(1, nClass);
        for iClass = 1:nClass
            recall(iClass) = label(:, iClass)'*predictedLabel(:, iClass)/sum(label(:, iClass));
        end
        recall(isnan(recall)) = 1;
    end

    function output = index2vector(input, nClass)
        output = zeros(numel(input), nClass);
        for iData = 1:numel(input)
            output(iData, input(iData)) = 1;
        end
    end
end
