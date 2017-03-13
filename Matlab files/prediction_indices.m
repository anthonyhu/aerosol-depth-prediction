function [indices] = prediction_indices(L_train, L, n)
%function [indices] = prediction_indices(L_train, L, n)
%Returns the indices of the predictions of L elements (=bags): indices{k}.train, indices{k}.val, indices{k}.test

%Copyright (C) 2012- Zoltan Szabo ("http://www.gatsby.ucl.ac.uk/~szabo/", "zoltan (dot) szabo (at) gatsby (dot) ucl (dot) ac (dot) uk")
%
%This file is part of the ITE (Information Theoretical Estimators) toolbox.
%
%ITE is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
%
%This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License along with ITE. If not, see <http://www.gnu.org/licenses/>.

L_train = floor(L_train/n) * n; %guarantee that n|L_train

%index_blocks:
    for k = 1 : n
        index_blocks{k} = [(k-1)*L_train/n+1:k*L_train/n];
    end
    
indices = {};
%for k = 1 : n
k = 1;
    %idx_train, idx_val, idx_test:
        idx_train = [1:n-1] + (k-1);
        idx_val = [n] + (k-1);
    %correct the interval if its elements are out of [1:n]    
        idx_train = correct_interval(idx_train,n);
        idx_val = correct_interval(idx_val,n);

        %idx_test = correct_interval(idx_test,n);

        % Define test index
        idx_test = [(L_train + 1):L];
    indices{1}.train = union2(index_blocks(idx_train));
    indices{1}.val = union2(index_blocks(idx_val));
    indices{1}.test = idx_test;
end