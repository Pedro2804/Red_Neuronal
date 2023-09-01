function [x] = full_conv(in,w)
    zin = size(in);
    zw  = size(w);
    %wr  = 0.*w;
    %x   = zeros(zin(1)+zw(1)-1,zin(2)+zw(2)-1);
    
%     %rotar w
%     for kx=1:1:zw(1)
%         for ky=1:1:zw(2)
%             wr(kx,ky) = w(zw(1)+1-kx,zw(2)+1-ky);
%         end
%     end
    in_aux = zeros(zin(1)+(zw(1)-1)*2,zin(2)+(zw(2)-1)*2);
    in_aux(zw(1):zw(1)-1+zin(1),zw(2):zw(2)-1+zin(2)) = in;
    x = conv2(in_aux,w,'valid');
    %x = conv2(in_aux,rot90(rot90(w)),'valid');
end