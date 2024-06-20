%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description:
%           LCS_P fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by
%           exploiting local based Pan simulation and golabl injection model
% Interface:
%           I_Fus_LCS_P = LCS_P(I_MS,I_PAN,I_MS_LR,ratio)
%
% Inputs:
%           I_MS:       MS image upsampled at PAN scale;
%           I_PAN:      PAN image;
%           I_MS_LR:    MS image;
%           ratio:      Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           LCS_P:  LCS_P pasharpened image.
%
% References:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_LCS_P = LCS_P(I_MS,I_PAN,ratio)
%% 通过引导滤波，合成低分辨率全色影像 q
r = 4*ratio;%4*ratio;%根据影像的分辨率比值确定窗口宽度：2r+1
eps = 0;
[q,~,~] = guidedfilter_multibands(I_MS, I_PAN, r, eps);
%% 将全色的亮度范围调整到和合成的低分辨率影像一致（没有意义因为Pan 和 q 本身就非常接近）
I_PAN = (I_PAN - mean(I_PAN(:)))*std2(q)/std(I_PAN(:)) + mean2(q);
%% 基于引导滤波的成分代替融合(CR)：全局方法
%choice1：全局代替(CR)
I_LCS_P = zeros(size(I_MS));
dtails =(I_PAN-q);
dtails = dtails-mean(dtails(:));
for i = 1:size(I_MS,3)
    %按照合成的比例加回去
    Gweight = cov(q(:),I_MS(:,:,i))/var(q(:));
    I_LCS_P(:,:,i) = dtails.*Gweight(2) +I_MS(:, :, i);%尝试减去均值
end
end
