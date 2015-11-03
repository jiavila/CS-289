function Xcen = center_data(X)
%this function centers the data matrix X, assuming each row in the matrix
%is a sample and each column represents a feature.

%Get the expected value of each column of X and repeat it for every row
EX = repmat(mean(X), size(X,1),1);

Xcen = X -EX;
end
