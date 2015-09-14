function [split_data_cell, split_labels_cell, rand_inds_cell] = rand_split_data(data, labels, dim, k)
%This function randomly splits data into k equal parts.

%create a random index vector
rand_ind = randperm(numel(labels));

%split the rand_ind vector into k parts
a = floor(numel(labels)/k); %compute the number of elements for each split, minus the kth split
b = mod(numel(labels),k); %compute the remaining number of elements for the kth split

if b == 0
    c = repmat(a, 1, k);
else
    c = repmat(a, 1, k);%create cell for splitting rand_ind
    c(1:b) = c(1:b) + 1;
end

rand_inds_cell = mat2cell(rand_ind, 1, c);

split_data_cell = cell(size(rand_inds_cell));
split_labels_cell = cell(size(rand_inds_cell));
for ii = 1:1:numel(rand_inds_cell)
    %get the current random indeces
    curr_rands = rand_inds_cell{ii};
    
    %split the labels
    split_labels_cell{ii} = labels(curr_rands);
    
    %split the data
    if dim == 1
        split_data_cell{ii} = data(curr_rands, :);
    elseif dim == 2
        split_data_cell{ii} = data(:, curr_rands);
    elseif dim ==3
        split_data_cell{ii} = data(:,:,curr_rands);
    else
        
    end
end




end

