function [X,Y,W,B,Reg_pool] = layer_CNN(N,M,D,size_in,size_filter,reduction,gain)
    %N: Number of inputs
    %M: Number of channels
    
    ns = size_in-size_filter+1;
    
    X = zeros(size_in,size_in,D,N);
    Y = zeros(ns,ns,M,N);
    W = gain.*2.*rand(size_filter,size_filter,D,M)-gain.*1;
    B = gain.*2.*rand(M,1)-gain.*1;
    
    if reduction>0
        Reg_pool = zeros(size_in,size_in,3,D,N);
    end
end

