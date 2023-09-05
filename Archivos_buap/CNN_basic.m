if exist('ready_dataset')==0  %#ok
    view_dataset;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Developed by Carlos Leopoldo Carreón Díaz de León in 2022
%                                       
                                  % | %   
                                  % | %   
                                  % V %    

close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Variable initialization
LR	= 10e-3;        %Initial learning rate
MT	= 100;          %Loss function modifier for backpropagation algorithm
% c1	= 1.4e-4;       %Constant for adjusting the output layer of the CNN
c1	= 1e-5; 

test_set    = 0;    %Flag to select between training the CNN or only test 
                    %the CNN
sobre_train = 0;    %Flag to train the CNN from the actual kernels and 
                    %synaptic weights instead from random values


%Define the architecture of the CNN

%                     [Input: 28x28x1 size]
%                               |
%                               |
%                               V
%            [Conv layer 1: Kernel 9x9x2x10, bias 10x1, 
%           20x20x10 map size, ReLU activation function]
%                               |
%                               |
%                               V
%            [Conv layer 2: Kernel 5x5x10x10, bias 10x1, 
%            16x16x10 map size, ReLU activation function]
%                               |
%                               |
%                               V
%            [Conv layer 3: Kernel 3x3x10x10, bias 10x1, 
%           14x14x10 map size, ReLU activation function]
%                               |
%                               |
%                               V
%         [Rearrange the map of 14x14x10 to a vector of 1960x1]
%                               |
%                               |
%                               V
%       [Full connected layer 1: synaptic weights 100x1960, 
%                bias 100x1,Relu activation funcion]
%                               |
%                               |
%                               V
%       [Full connected layer 2: synaptic weights 100x100, 
%                bias 100x1,Relu activation funcion]
%                               |
%                               |
%                               V
%       [Full connected layer 3: synaptic weights 10x100, 
%                bias 10x1,sigmoid activation funcion]
%                               |
%                               |
%                               V
%                       [Output: 10x1 size]

if test_set==0
    if sobre_train==0
        cnn_D0 = 1;         cnn_M0 = 10;
        cnn_D1 = cnn_M0;    cnn_M1 = 10;
        cnn_D2 = cnn_M1;    cnn_M2 = 10;
        
        [X0,Y0,W0,B0]       = layer_CNN(1,    cnn_M0, cnn_D0, 28  ,9 ,0,1); %#ok<*ASGLU>
        [X1,Y1,W1,B1,R1]    = layer_CNN(1,    cnn_D1, cnn_M1, 20  ,5 ,2,1);
        [X2,Y2,W2,B2,R2]    = layer_CNN(1,    cnn_D2, cnn_M2, 16  ,3 ,2,1);
        
        [X3,Y3,W3,B3]       = layer_FCC(1960,100,1);
        [X4,Y4,W4,B4]       = layer_FCC(100,100,1);
        [X5,Y5,W5,B5]       = layer_FCC(100,10,1);
    end
end


%Storage variables
Z1 = zeros(28,28,1);      % Variable used for display the input image in 
                            % the training of the CNN
if test_set==0 
    % Iterations for training the CNN
    iteraciones = 500e3;
else
    % Iterations for test after the training the CNN
    iteraciones = 5e3;
end

E          = zeros(iteraciones,1);  % Loss function of the training set
Etest      = zeros(iteraciones,1);  % Loss function of the test set 
yCNN       = zeros(10,iteraciones);  % Output of the CNN 
yDPN       = zeros(10,iteraciones);  % Desired output 
sourc      = zeros(iteraciones,2);  % Input image source 






%Training of the CNN
for K = 1:iteraciones
    %LR = LR + 0.2000e-07;   % The learning rate LR increases 0.2000e-07 
                            % each iteration
    try_nan = 0;            %Flag to avoid not a number (NaN) values
    
    
    % Obtain the input image from the training set by random selection
    while try_nan<1
        sp  = floor(rand(1,1)*(10e3-1))+1;
        X0 = Amnist(:,:,sp);
        % YD = (1+exp(-ZP_lab(:,sp))).^-1;    

        yd  = label_A(sp);
        switch(yd)
            case 0; yd_trt = [1,0,0,0,0,0,0,0,0,0]';
            case 1; yd_trt = [0,1,0,0,0,0,0,0,0,0]';
            case 2; yd_trt = [0,0,1,0,0,0,0,0,0,0]';
            case 3; yd_trt = [0,0,0,1,0,0,0,0,0,0]';
            case 4; yd_trt = [0,0,0,0,1,0,0,0,0,0]';
            case 5; yd_trt = [0,0,0,0,0,1,0,0,0,0]';
            case 6; yd_trt = [0,0,0,0,0,0,1,0,0,0]';
            case 7; yd_trt = [0,0,0,0,0,0,0,1,0,0]';
            case 8; yd_trt = [0,0,0,0,0,0,0,0,1,0]';
            case 9; yd_trt = [0,0,0,0,0,0,0,0,0,1]';
        end
        % YD = 1./(1+exp(-  (2.*yd_trt-1)   ));
        YD = yd_trt;

        if isnan(X0)==0
            try_nan=1;
        else
            keep1 = sp;
        end
    end
    
    try_nan = 0;
    % Obtain the input image from the test set by random selection
    while try_nan<1
        sp_test  = floor(rand(1,1)*(59999-1))+1;
        X0_test  = Bmnist(:,:,sp_test);
        % YD_test  = (1+exp(-ZP_lab(:,sp_test))).^-1; 

        yd  = label_B(sp_test);
        switch(yd)
            case 0; yd_trt = [1,0,0,0,0,0,0,0,0,0]';
            case 1; yd_trt = [0,1,0,0,0,0,0,0,0,0]';
            case 2; yd_trt = [0,0,1,0,0,0,0,0,0,0]';
            case 3; yd_trt = [0,0,0,1,0,0,0,0,0,0]';
            case 4; yd_trt = [0,0,0,0,1,0,0,0,0,0]';
            case 5; yd_trt = [0,0,0,0,0,1,0,0,0,0]';
            case 6; yd_trt = [0,0,0,0,0,0,1,0,0,0]';
            case 7; yd_trt = [0,0,0,0,0,0,0,1,0,0]';
            case 8; yd_trt = [0,0,0,0,0,0,0,0,1,0]';
            case 9; yd_trt = [0,0,0,0,0,0,0,0,0,1]';
        end
        % YD_test = 1./(1+exp(-  (2.*yd_trt-1)   ));
        YD_test = yd_trt;
        if isnan(X0_test)==0
            try_nan=1;
        else
            keep1 = sp_test;
        end
    end
    
    sourc(K,1) = sp;
    sourc(K,2) = sp_test;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                         Test run of the CNN
    %
    for km = 1:cnn_M0
        sm1 = 0.*Y0(:,:,1);
        for kd = 1:cnn_D0
            %-----------------------------------------------%
            am1 = zeros(9,9);                               %
            for q1 = 1:9                                    %
                for q2 = 1:9                                %
                    am1(q1,q2) = W0(9-q1+1,9-q2+1,kd,km);   %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            sm1 = sm1 + conv2(X0_test(:,:,kd),am1,'valid');
        end
        Y0(:,:,km) = max(sm1 + B0(km),0);
        X1(:,:,km) = Y0(:,:,km);
        % [X1(:,:,km),R1(:,:,:,km)] = max_pool(Y0(:,:,km),2);
    end

    for km = 1:cnn_M1
        sm1 = 0.*Y1(:,:,1);
        for kd = 1:cnn_D1
            %-----------------------------------------------%
            am1 = zeros(5,5);                               %
            for q1 = 1:5                                    %
                for q2 = 1:5                                %
                    am1(q1,q2) = W1(5-q1+1,5-q2+1,kd,km);   %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            sm1 = sm1 + conv2(X1(:,:,kd),am1,'valid');
        end
        Y1(:,:,km) = max(sm1 + B1(km),0);
        X2(:,:,km) = Y1(:,:,km);
        % [X2(:,:,km),R2(:,:,:,km)] = max_pool(Y1(:,:,km),2);
    end
    for km = 1:cnn_M2
        sm2 = 0.*Y2(:,:,1);
        for kd = 1:cnn_D2
            %-----------------------------------------------%
            am2 = zeros(3,3);                               %
            for q1 = 1:3                                    %
                for q2 = 1:3                                %
                    am2(q1,q2) = W2(3-q1+1,3-q2+1,kd,km);   %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            sm2 = sm2 + conv2(X2(:,:,kd),am2,'valid');
        end
        Y2(:,:,km) = max(sm2 + B2(km),0);
    end
    X3 = reshape(Y2,[],1);

    X3g = gpuArray(X3);
    W3g = gpuArray(W3);
    Y3g = pagefun(@mtimes,W3g,X3g);
    Y3  = gather(Y3g);
    Y3  = max(Y3 + 1.*B3,0);
    
    X4  = Y3;
    X4g = gpuArray(X4);
    W4g = gpuArray(W4);
    Y4g = pagefun(@mtimes,W4g,X4g);
    Y4  = gather(Y4g);
    Y4  = max(Y4 + 1.*B4,0);
    
    X5 = Y4;
    X5g = gpuArray(X5);
    W5g = gpuArray(W5);
    Y5g = pagefun(@mtimes,W5g,X5g);
    Y5  = gather(Y5g);
    
    
    % Y5       = (1+exp(c1.*(-Y5-B5))).^-1;
    Y5 = exp(c1.*(Y5+B5))./sum(exp(c1.*(Y5+B5)));
    Etest(K) = 0.5*mean( (YD_test-Y5).^2 );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                         Test run of the CNN
    %
    for km = 1:cnn_M0
        sm1 = 0.*Y0(:,:,1);
        for kd = 1:cnn_D0
            %-----------------------------------------------%
            am1 = zeros(9,9);                               %
            for q1 = 1:9                                    %
                for q2 = 1:9                                %
                    am1(q1,q2) = W0(9-q1+1,9-q2+1,kd,km);   %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            sm1 = sm1 + conv2(X0(:,:,kd),am1,'valid');
        end
        Y0(:,:,km) = max(sm1 + B0(km),0);
        X1(:,:,km) = Y0(:,:,km);
        % [X1(:,:,km),R1(:,:,:,km)] = max_pool(Y0(:,:,km),2);
    end

    for km = 1:cnn_M1
        sm1 = 0.*Y1(:,:,1);
        for kd = 1:cnn_D1
            %-----------------------------------------------%
            am1 = zeros(5,5);                               %
            for q1 = 1:5                                    %
                for q2 = 1:5                                %
                    am1(q1,q2) = W1(5-q1+1,5-q2+1,kd,km);   %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            sm1 = sm1 + conv2(X1(:,:,kd),am1,'valid');
        end
        Y1(:,:,km) = max(sm1 + B1(km),0);
        X2(:,:,km) = Y1(:,:,km);
        % [X2(:,:,km),R2(:,:,:,km)] = max_pool(Y1(:,:,km),2);
    end
    
    for km = 1:cnn_M2
        sm2 = 0.*Y2(:,:,1);
        for kd = 1:cnn_D2
            %-----------------------------------------------%
            am2 = zeros(3,3);                               %
            for q1 = 1:3                                    %
                for q2 = 1:3                                %
                    am2(q1,q2) = W2(3-q1+1,3-q2+1,kd,km);   %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            sm2 = sm2 + conv2(X2(:,:,kd),am2,'valid');
        end
        Y2(:,:,km) = max(sm2 + B2(km),0);
    end
    X3 = reshape(Y2,[],1);

    X3g = gpuArray(X3);
    W3g = gpuArray(W3);
    Y3g = pagefun(@mtimes,W3g,X3g);
    Y3  = gather(Y3g);
    Y3  = max(Y3 + 1.*B3,0);
    
    X4  = Y3;
    X4g = gpuArray(X4);
    W4g = gpuArray(W4);
    Y4g = pagefun(@mtimes,W4g,X4g);
    Y4  = gather(Y4g);
    Y4  = max(Y4 + 1.*B4,0);
    
    X5 = Y4;
    X5g = gpuArray(X5);
    W5g = gpuArray(W5);
    Y5g = pagefun(@mtimes,W5g,X5g);
    Y5  = gather(Y5g);
    
    
    % Y5   = (1+exp(c1.*(-Y5-B5))).^-1;
    Y5_past = Y5;
    Y5 = exp(c1.*(Y5+B5))./sum(exp(c1.*(Y5+B5)));
    
    % if sl==1
        YD_neg = YD;
        Y5_neg = Y5;
    % end
    
    E(K) = 0.5*mean( (YD-Y5).^2 );
    yCNN(:,K) = Y5;
    yDPN(:,K) = YD;
    
    % Visualization of the training process
    if mod(K-1,1e3)==999
        Q1 = E(K-999:K);
        Q2 = Etest(K-999:K);
        subplot(1,2,1)
        semilogy(K,mean(Q1),'b.',K,mean(Q2),'r.')%,K,Last_error,'r.')
        % [YD_neg,Y5_neg,abs(YD_neg-Y5_neg)]
        axf = find(Y5_neg==max(Y5_neg)) - 1;
        switch(axf)
            case 0; mxmp = [1,0,0,0,0,0,0,0,0,0]';
            case 1; mxmp = [0,1,0,0,0,0,0,0,0,0]';
            case 2; mxmp = [0,0,1,0,0,0,0,0,0,0]';
            case 3; mxmp = [0,0,0,1,0,0,0,0,0,0]';
            case 4; mxmp = [0,0,0,0,1,0,0,0,0,0]';
            case 5; mxmp = [0,0,0,0,0,1,0,0,0,0]';
            case 6; mxmp = [0,0,0,0,0,0,1,0,0,0]';
            case 7; mxmp = [0,0,0,0,0,0,0,1,0,0]';
            case 8; mxmp = [0,0,0,0,0,0,0,0,1,0]';
            case 9; mxmp = [0,0,0,0,0,0,0,0,0,1]';
        end

        [YD_neg,mxmp,abs(YD_neg-mxmp)]%#ok
        LR                          % Display the dots of the loss 
                                    % function, the desired value and 
                                    % CNN value of the iteration and 
                                    % the learning rate
        
        hold on
        subplot(1,2,2)
        imshow(X0.*0.5);
        pause(1e-20);
    end
    
    % Back propagation error
    if test_set==0
        dE5 = (Y5-YD).*MT;
        dF5 = c1.*Y5.*(1-Y5);

        dC5 = dE5.*dF5;
        dC5g= gpuArray(dC5);
        dW5 = -LR.*X5'.*dC5;
        dB5 = -LR.*dC5;
        %%%
        dE4g = pagefun(@mtimes,W5g',dC5g);
        dE4  = gather(dE4g);
        dF4 = sign(Y4);
        dC4 = dE4.*dF4;

        dC4g= gpuArray(dC4);

        dW4g= pagefun(@mtimes,dC4g,X4g');
        dW4 = gather(dW4g);
        dW4 = bsxfun(@times,-LR,dW4);
        dB4 = bsxfun(@times,-LR,dC4);   
        %%%
        dE3g = pagefun(@mtimes,W4g',dC4g);
        dE3  = gather(dE3g);
        dF3 = sign(Y3);
        dC3 = dE3.*dF3;

        dC3g= gpuArray(dC3);

        dW3g= pagefun(@mtimes,dC3g,X3g');
        dW3 = gather(dW3g);
        dW3 = bsxfun(@times,-LR,dW3);
        dB3 = bsxfun(@times,-LR,dC3);   
        %%%
        dE2fg = pagefun(@mtimes,W3g',dC3g);
        dE2f  = gather(dE2fg); 

        dE2  = reshape(dE2f,14,14,10);
        dF2  = sign(Y2);
        dC2  = dE2.*dF2;

        dW2  = 0.*W2;
        dB2  = 0.*B2;

        for km = 1:cnn_M2
            %-----------------------------------------------%
            dCs2 = zeros(14,14);                            %
            for q1 = 1:14                                   %
                for q2 = 1:14                               %
                    dCs2(q1,q2) = dC2(14-q1+1,14-q2+1,km);  %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            for kd = 1:cnn_D2
                dW2(:,:,kd,km) = -LR.*conv2(X2(:,:,kd),dCs2,'valid');
            end
            dB2(km) = -LR.*sum(sum(dCs2));
        end
        %%%
        dE1p = 0.*X2;
        for kd = 1:cnn_D2
            aq1 = 0.*dE1p(:,:,1);
            for km = 1:cnn_M2
                aq1 = aq1 + full_conv(dC2(:,:,km),W2(:,:,kd,km));
            end
            dE1p(:,:,kd) = aq1;
        end

        dE1 = dE1p;
%         dE1 = 0.*Y1;
%         for xq = 1:21
%             for yq = 1:21
%                 for d = 1:10
%                     label = R2(xq,yq,3,d);
%                     xp    = R2(xq,yq,1,d);
%                     yp    = R2(xq,yq,2,d);
%                     if label==1
%                         dE1(xp,yp,d) = dE1p(xq,yq,d);
%                     end
%                 end
%             end
%         end
        dF1 = sign(Y1);
        dC1 = dE1.*dF1;

        dW1  = 0.*W1;
        dB1  = 0.*B1;

        for km = 1:cnn_M1
            %-----------------------------------------------%
            dCs1 = zeros(16,16);                            %
            for q1 = 1:16                                   %
                for q2 = 1:16                               %
                    dCs1(q1,q2) = dC1(16-q1+1,16-q2+1,km);  %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            for kd = 1:cnn_D1
                dW1(:,:,kd,km) = -LR.*conv2(X1(:,:,kd),dCs1,'valid');
            end
            dB1(km) = -LR.*sum(sum(dCs1));
        end
        %%%
        dE0p = 0.*X1;
        for kd = 1:cnn_D1
            aq0 = 0.*dE0p(:,:,1);
            for km = 1:cnn_M1
                aq0 = aq0 + full_conv(dC1(:,:,km),W1(:,:,kd,km));
            end
            dE0p(:,:,kd) = aq0;
        end

        dE0 = dE0p;
%         dE0 = 0.*Y0;
%         for xq = 1:46
%             for yq = 1:46
%                 for d = 1:10
%                     label = R1(xq,yq,3,d);
%                     xp    = R1(xq,yq,1,d);
%                     yp    = R1(xq,yq,2,d);
%                     if label==1
%                         dE0(xp,yp,d) = dE0p(xq,yq,d);
%                     end
%                 end
%             end
%         end
        dF0 = sign(Y0);
        dC0 = dE0.*dF0;

        dW0  = 0.*W0;
        dB0  = 0.*B0;

        for km = 1:cnn_M0
            %-----------------------------------------------%
            dCs0 = zeros(20,20);                            %
            for q1 = 1:20                                   %
                for q2 = 1:20                               %
                    dCs0(q1,q2) = dC0(20-q1+1,20-q2+1,km);  %
                end                                         %
            end                                             %
            %-----------------------------------------------%
            for kd = 1:cnn_D0
                dW0(:,:,kd,km) = -LR.*conv2(X0(:,:,kd),dCs0,'valid');
            end
            dB0(km) = -LR.*sum(sum(dCs0));
        end
        
        if isnan(dW0)==1
            disp('NaN');
            break;
        end
        
        if isnan(dW1)==1
            disp('NaN');
            break;
        end
        
        if isnan(dW2)==1
            disp('NaN');
            break;
        end
        
        if isnan(dW3)==1
            disp('NaN');
            break;
        end
        
        if isnan(dW4)==1
            disp('NaN');
            break;
        end
        
        if isnan(dW5)==1
            disp('NaN');
            break;
        end
        
        W5 = W5 + dW5;
        B5 = B5 + dB5;

        W4 = W4 + dW4;
        B4 = B4 + dB4;

        W3 = W3 + dW3;
        B3 = B3 + dB3;

        W2 = W2 + dW2;
        B2 = B2 + dB2;

        W1 = W1 + dW1;
        B1 = B1 + dB1;

        W0 = W0 + dW0;
        B0 = B0 + dB0;
        

    end
end