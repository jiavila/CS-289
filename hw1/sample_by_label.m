function [sams, sam_labels, remain_data, remain_labels, I, uniq_labels] = sample_by_label(data, labels, N)
%This function gets samples from the differently-labeled data.  The
%separated data is evenly sampled.
%
%             [sep_data, sep_labels, I] = sample_by_label(data, labels, num_sam_per_label)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - INPUTS - %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%data               MxNxL or MxN        The input data
%
%labels             Mx1                 The labels
%
%num_sam_per_label  1x1 double/int      The total number of samples to 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - OUTPUTS - %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sep_data                               
%
%sep_labels
%
%I
%
%uniq_labels



%Get the number of unique labels
uniq_labels = unique(labels);

num_sam_per_label = ceil(N/numel(uniq_labels));

%create our logical output
I = false(size(labels));

for ii = 1:1:numel(uniq_labels)
    %Find all instances of the current lable, returns logical
    I_temp = labels == uniq_labels(ii);
    
    %get the indeces of the non-zero logicals
    inds = find(I_temp);
    
    %get the first x indeces from I_temp
    xth_ind = inds(num_sam_per_label);
    inds_sep = false(size(labels));
    inds_sep(1:xth_ind) = I_temp(1:xth_ind);
    
    %Store the separated inds in I
    I = I | inds_sep;
end
    
%Separate the data according to our final I
if num(size(data3)) == 3
    sams = data(:,:,I);
    remain_data = data(:,:,~I);
else
    sams = data(I,:);
    remain_data = data(~I,:);
end
sam_labels = labels(I);

remain_labels = labels(~I);





end

