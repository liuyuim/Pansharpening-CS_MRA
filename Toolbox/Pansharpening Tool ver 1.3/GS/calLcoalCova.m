function [ cov_12 ] = calLcoalCova(band1, band2,r)
%CALLCOALCOVA calculate  covariance of (band1, band) in each local patch.
%   band1:the first input band 
%   bnad2:the second input band
%   r: the raduis of the square window
% colfilt(q,[2*R+1 2*R+1],'sliding',@var)
if ~isequal(size(band1),size(band2))
    error('输入影像的必须大小相等！')
end
if size(band1,3) ~= 1
    error('输入影像的必须为单波段影像！')
end
if size(band2,3) ~= 1
    error('输入影像的必须为单波段影像！')
end

[hei, wid,~] = size(band1);
N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.

mean_1 = boxfilter(band1, r) ./ N;
mean_2 = boxfilter(band2, r) ./ N;
mean_12 = boxfilter(band1.*band2, r) ./ N;

% covariance of (I, p) in each local patch.
cov_12 =  mean_12-mean_1.*mean_2;
end

