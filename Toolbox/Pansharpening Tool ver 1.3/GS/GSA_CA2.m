%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           GSA_CA2 fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Gram-Schmidt Adaptive_Context adaptive(GSA_CA2) algorithm.
% 
% Interface:
%           I_Fus_GSA_CA = GSA_CA2(I_MS,I_PAN,I_MS_LR,ratio)
%
% Inputs:
%           I_MS:       MS image upsampled at PAN scale;
%           I_PAN:      PAN image;
%           I_MS_LR:    MS image;
%           ratio:      Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_GSA:  GSA pasharpened image.
% 
% References:
%           [Aiazzi09]  A Comparison Between Global and Context-Adaptive Pansharpening of Multispectral Images
%           [Vivone14]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, 揂 Critical Comparison Among Pansharpening Algorithms? 
%                       IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
% 在GSA函数的基础上修改，将全局的注入模型更改为局部自适应窗口
% 同时对局部系数进行平均
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_Fus_GSA = GSA_CA2(I_MS,I_PAN,I_MS_LR,ratio)

imageLR = double(I_MS);
imageHR = double(I_PAN);
imageLR_LP = double(I_MS_LR);

%%% Remove means from imageLR
imageLR0 = zeros(size(I_MS));
for ii = 1 : size(I_MS,3), imageLR0(:,:,ii) = imageLR(:,:,ii) - mean2(imageLR(:,:,ii)); end

%%% Remove means from imageLR_LP
imageLR_LP0 = zeros(size(I_MS_LR));
for ii = 1 : size(I_MS_LR,3), imageLR_LP0(:,:,ii) = imageLR_LP(:,:,ii) - mean2(imageLR_LP(:,:,ii)); end


%% Intensity
imageHR0 = imageHR - mean2(imageHR);
imageHR0 = LPfilterPlusDec(imageHR0,ratio);
alpha(1,1,:) = estimation_alpha(cat(3,imageLR_LP0,ones(size(I_MS_LR,1),size(I_MS_LR,2))),imageHR0,'global');
I = sum(cat(3,imageLR0,ones(size(I_MS,1),size(I_MS,2))) .* repmat(alpha,[size(I_MS,1) size(I_MS,2) 1]),3); 

%%% Remove mean from I
I0 = I - mean2(I);

%% Coefficients
[hei, wid, dim]= size(I_MS);

gm = ones(hei, wid, dim+1);
r = 4*ratio;%处理半径为2r+1
N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
for i = 1:dim  
        %     %M5:局部GS方法:将差异归一化（减去均值）再加入到原始影像中
    weight = calLcoalCova(I0,I_MS(:,:,i),r)./(calLcoalCova(I0,I0,r));
    gm(:,:,i+1) = boxfilter(weight, r)./N;
end
gm = reshape(gm,[hei*wid, dim+1]);

imageHR = imageHR - mean2(imageHR);

%%% Detail Extraction
delta = imageHR - I0;
deltam = repmat(delta(:),[1 size(I_MS,3)+1]);

%% Fusion
V = I0(:);
for ii = 1 : size(I_MS,3)
    h = imageLR0(:,:,ii);
    V = cat(2,V,h(:));
end

V_hat = V + deltam .* gm;

%%% Reshape fusion result
I_Fus_GSA = reshape(V_hat(:,2:end),[size(I_MS,1) size(I_MS,2) size(I_MS,3)]);

%%% Final Mean Equalization
for ii = 1 : size(I_MS,3)
    h = I_Fus_GSA(:,:,ii);
    I_Fus_GSA(:,:,ii) = h - mean2(h) + mean2(imageLR(:,:,ii));
end

end