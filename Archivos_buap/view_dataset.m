% Read dataset
load('Amnist.mat')
% load('Bmnist.mat')
load('index_A.mat')
% load('index_B.mat')
load('Bmnist_V2');
load('index_B_V2');

label_A                                         = zeros(10e3,1);
label_A(1:index_A(1))                           = 0;
label_A(index_A(1)+1:sum(index_A(1:2)))         = 1;
label_A(sum(index_A(1:2))+1:sum(index_A(1:3)))  = 2;
label_A(sum(index_A(1:3))+1:sum(index_A(1:4)))  = 3;
label_A(sum(index_A(1:4))+1:sum(index_A(1:5)))  = 4;
label_A(sum(index_A(1:5))+1:sum(index_A(1:6)))  = 5;
label_A(sum(index_A(1:6))+1:sum(index_A(1:7)))  = 6;
label_A(sum(index_A(1:7))+1:sum(index_A(1:8)))  = 7;
label_A(sum(index_A(1:8))+1:sum(index_A(1:9)))  = 8;
label_A(sum(index_A(1:9))+1:sum(index_A(1:10))) = 9;


% label_B                                         = zeros(59999,1);
% label_B(1:index_B(1))                           = 0;
% label_B(index_B(1)+1:sum(index_B(1:2)))         = 1;
% label_B(sum(index_B(1:2))+1:sum(index_B(1:3)))  = 2;
% label_B(sum(index_B(1:3))+1:sum(index_B(1:4)))  = 3;
% label_B(sum(index_B(1:4))+1:sum(index_B(1:5)))  = 4;
% label_B(sum(index_B(1:5))+1:sum(index_B(1:6)))  = 5;
% label_B(sum(index_B(1:6))+1:sum(index_B(1:7)))  = 6;
% label_B(sum(index_B(1:7))+1:sum(index_B(1:8)))  = 7;
% label_B(sum(index_B(1:8))+1:sum(index_B(1:9)))  = 8;
% label_B(sum(index_B(1:9))+1:sum(index_B(1:10))) = 9;
Bmnist = Bmnist_V2;
clear('Bmnist_V2');
label_B = index_B_V2;
clear('index_B_V2');

ready_dataset = 1;
