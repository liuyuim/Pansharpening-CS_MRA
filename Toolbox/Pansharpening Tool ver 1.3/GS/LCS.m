%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           LCS fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting local based Pan simulation and lcoal injection
% 
% Interface:
%           I_Fus_LCS_P = LCS(I_MS,I_PAN,I_MS_LR,ratio)
%
% Inputs:
%           I_MS:       MS image upsampled at PAN scale;
%           I_PAN:      PAN image;
%           ratio:      Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           LCS:  LCS pasharpened image.
% 
% References:
%           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_LCS = LCS(I_MS,I_PAN,ratio)
%% 通过引导滤波，合成低分辨率全色影像 q
r = 4*ratio;%根据影像的分辨率比值确定窗口宽度：2r+1
eps = 0;
[q,~,~] = guidedfilter_multibands(I_MS, I_PAN, r, eps);
%% 将全色的亮度范围调整到和合成的低分辨率影像一致（没有意义因为Pan 和 q 本身就非常接近）
I_PAN = (I_PAN - mean(I_PAN(:)))*std2(q)/std(I_PAN(:)) + mean2(q);
%% 基于引导滤波的成分代替融合(CR)：局部方法
dtails =(I_PAN-q);
% dtails = dtails-mean(dtails(:));%是否减去均值，对最终结果影响不大

I_LCS = zeros(size(I_MS));
[hei, wid, ~]= size(I_MS);
N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
for i = 1:size(I_MS,3)    
        %     %M5:局部GS方法:将差异归一化（减去均值）再加入到原始影像中
    weight = calLcoalCova(q,I_MS(:,:,i),r)./(calLcoalCova(q,q,r));
    weight = boxfilter(weight, r)./N;
    I_LCS(:,:,i) = dtails.*weight +I_MS(:, :, i);  
    %仅仅该步骤无助于融合精度的改善
%      I_LCS(:,:,i) = I_LCS(:,:,i)-mean2(I_LCS(:,:,i))+mean2(I_MS(:, :,i));
%      
end
end
