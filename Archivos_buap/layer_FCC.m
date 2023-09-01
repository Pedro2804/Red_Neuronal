function [X,Y,W,B] = layer_FCC(in,out,gain)
    X = zeros(in,1);
    Y = zeros(out,1);
    W = gain.*2.*rand(out,in)-gain;
    B = gain.*2.*rand(out,1)-gain;
end

