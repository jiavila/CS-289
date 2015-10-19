function [split_data_cell, split_labels_cell, rand_inds_cell] = rand_split_data(data, labels, dim, k)
%This function randomly splits data into k equal parts.

%create an index vector
I = (1:1:numel(labels))';

%Separate the index vector by labels

%Get the number of unique labels
uniq_labels = unique(labels);

rand_inds_cell = cell(k, numel(uniq_labels));
for ii = 1:1:numel(uniq_labels)
    %Find all instances of the current lable, returns logical
    I_temp = labels == uniq_labels(ii);
    
    %get the corresponding indeces from I
    I_temp = I(I_temp);
    
    %randomly permute the indeces
    I_temp = I_temp(randperm(numel(I_temp)));
    
    %Split the vector into k-part cell column
    a = floor(numel(I_temp)/k); %compute the number of elements for each split, minus the kth split
    b = mod(numel(I_temp),k); %compute the remaining number of elements for the kth split
    
    if b == 0
        c = repmat(a, 1, k);
    else
        c = repmat(a, 1, k);%create cell for splitting rand_ind
        c(1:b) = c(1:b) + 1;
    end
    
    I_temp = mat2cell(I_temp, c, 1);
    
    %store in our cell along the first row
    rand_inds_cell(:, ii) = I_temp;
end

%Transpose rand_inds_cell to stack each k group along the columns of the
%cell
rand_inds_cell = rand_inds_cell';

%Concatenate the columns to create the final indeces
for ii = 1:1:k
    I_temp = cat(1, rand_inds_cell{:, ii});
    
    %store it in the first row
    rand_inds_cell{1,ii} = I_temp;
    
    %empty rows 2 through end
    rand_inds_cell(2:end,ii) = cell(size(rand_inds_cell, 1) - 1,1);
    
    
end

%remove the empty cells of rand_inds_cell
rand_inds_cell = rand_inds_cell(1,:);

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

