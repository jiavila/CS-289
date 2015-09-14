function [ A_norm ] = norm_row(A)
%This function normalizes a matrix along the row dimension

%compute the magnitude of the rows
normVec = (sum(A.*A, 2)).^.5;

%repeat the vector along the column dimension for division
normMat = repmat(normVec, [1 size(A, 2)]);

A_norm = A./normMat;
end

