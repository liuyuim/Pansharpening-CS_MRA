%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 由 Demo_for share0327\Demo_FusionComparsions.m （见结尾附录）修改
% 
% 使用Demo_CreateFusionPairs生成的mat文件包括如下结构
% |--(4)MS 256*256*4 (对应训练集——> gt 64 64 8)
% |--(2)MS_LR 64*64*4 (对应训练集——> ms 16 16 8) 
% |--(3)MS_LR_Up 256*256*4 (对应训练集——> lms 64 64 8)
% |--(6)MS_UP 1024*1024*4
% |--(5)PAN 1024*1024
% |--(1)PAN_LR 256*256 (对应训练集——> pan 64 64)
% |--Paras 1*1
% 降分辨率 监督评价
%                (1)PAN_LR   |
%                            |-- (4)MS / GT
% (2)MS_LR  ——>  (3)MS_LR_Up |    
% 
% 全分辨率 非监督评价
%             (5)PAN    |
%                       |-- ?
% (4)MS  ——>  (6)MS_UP  |
% 
% 在本程序中 全分辨率 非监督评价 是将 (5)PAN 和 (6)MS_UP 进行融合
% 降分辨率 监督评价 是将 (1)PAN_LR 和 (3)MS_LR_Up 进行融合

% run( 'file1.m' ) % 运行第一个脚本文件
% run( 'file2.m' ) % 运行第二个脚本文件
% 值得一提的是：这些个脚本文件可以在不同的文件夹下。比如一个在D盘某个文件夹下，一个在F盘某个文件夹下。
% 这时，只需要在run()语句的字符串中加入脚本文件的绝对路径即可。脚本文件执行时，也不会受到当前路径的影响。
% 例如：run( 'D:\file1.m' ); 下面如果还有其他脚本文件，则都可以如法炮制
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all; addpath(genpath('.\Fx\'));

% GF1_GengDi
% clear
% ImgPaths = 'G:\AFusionGroup\...\GF1_GengDi\'; %Fusion数据所在路径
% saveDir = 'G:\AFusionGroup\...\ChuantongResult\'; %设置对应保存路径
% FusionComparsions (ImgPaths, saveDir); %运行局部函数

% ImgPaths = 'F:\AFusionGroup\Data_Pairs_Ground\GF1_JianShe\'; %Fusion数据所在路径
% saveDir = 'F:\AFusionGroup\ShiyanLiuYuan\ChuantongResult\GF1_JianShe2\'; %设置对应保存路径
% FusionComparsions (ImgPaths, saveDir);

SensorNames = { 'GF1' }; % 'GF1','IK','QB','WV2','WV3','WV4'   'GF1','GF2','IK','JL1','QB','WV2','WV3','WV4'
for i = 1:numel(SensorNames)
    LandTypeNames = {'Data'};  % 'Data' 'GengDi','JianShe'
    for j = 1:numel(LandTypeNames)
        Sensor_Data = strcat(SensorNames{i},'_',LandTypeNames{j});    % 或者    Sensor_Data = SensorNames{i} + "_Data";    
        
%         ImgPaths = fullfile('F:\AFusionGroup\Shiyan\shiyan20231231\Benchmark\',Sensor_Data,'\');
        ImgPaths = fullfile('F:\Demo\Data_CSMRA\',Sensor_Data,'\Benchmark\');
        saveDir = fullfile('F:\Demo\Data_CSMRA\',Sensor_Data,'\CSMRA_Result\');    
        FusionComparsions (ImgPaths, saveDir); %运行局部函数
    end    
end

%%
% 将MatrixImage矩阵中的图片保存成图片
clc; clear; close all; addpath(genpath('.\Fx\'));

Path = 'F:\HC550WDC16TO\Shiyan\20231203\WV3_Data\CS-MRA_Result\39.mat';
SaveDir = 'F:\HC550WDC16TO\Shiyan\20231203\WV3_Data\CS-MRA_Result\Imwrite_mat39\';
MatrixImage2Tif(Path, SaveDir); % MatrixImage2Imwrite (Path, SaveDir)

% SensorNames = {'GF1','GF2','IK','JL1','QB','WV2','WV3','WV4'}; % SensorNames = {'GF1','GF2','IK','JL1','QB','WV2','WV3','WV4'}; 

% for i = 1:numel(SensorNames)
%     Sensor_Data = strcat(SensorNames{i}, '_Data');    % 或者    Sensor_Data = SensorNames{i} + "_Data";    
%     for panSize = [1024,512,256,128,64,32] 
%         Path = 'F:\AFusionGroup\ShiyanLiuYuan\ChuantongResult\GF1_GengDi\90.mat';
%         SaveDir = 'F:\AFusionGroup\ShiyanLiuYuan\ChuantongResult\GF1_GengDi\Imwrite_mat90\';
%         MatrixImage2Imwrite (Path, SaveDir);
%     end
% end


%%
% 创建包含局部函数的脚本local functions in scripts
function [] = FusionComparsions (ImgPaths, saveDir)
    
    % 采用benchmark数据集进行融合实验，在高低两个分辨率尺度上对融合结果进行评价
    % 对比方法采用 Pansharpening Tool ver 1.3中的方法；
    % 对比指标主要是
    

    % addpath(genpath('./Toolbox\'));
        
    if ~exist(saveDir,'dir')%待保存的图像文件夹不存在，就建文件夹
        mkdir(saveDir)
    end
        
    %列出传感器文件夹内所有的融合
    listing = dir([ImgPaths,'*.mat']) ;
    NumImgs = size(listing,1);
    % MatrixResults_Fu = zeros(19, 5, NumImgs);%存储融合结果的矩阵
    % MatrixResults_DR = zeros(19, 5, NumImgs);%存储j降分辨率融合结果的矩阵
    MatrixResults_Fu = [];MatrixResults_DR = []; 
    for i = 1:NumImgs
                    
        loadImgPath = [listing(i).folder,'\',listing(i).name];
        imgData = load(loadImgPath);
        
        formatSpec = '处理该二级目录%d个图像中第%d个！%s \n';
        fprintf(formatSpec, NumImgs, i, loadImgPath);

        %Full resoution results
        I_MS_LR = double(imgData.MS); % MS image;
        I_MS =  double(imgData.MS_Up);% MS image upsampled to the PAN size;
        I_PAN = double(imgData.Pan); %Pan
        Params = imgData.Paras;
        [MatrixImage_Fu, MatrixResult_Fu] = FusionAndEvaluateOnFullResolution (I_MS, I_MS_LR, I_PAN, Params);
        MatrixResults_Fu = cat(3, MatrixResults_Fu, MatrixResult_Fu);
        
        %Reduced resoution results
        I_GT = double(imgData.MS); %ground truth
        I_PAN = double(imgData.Pan_LR);% low resolution Pan image
        I_MS  = double(imgData.MS_LR_Up);% low resolution MS image upsampled at  low resolution  PAN scale;
        I_MS_LR = double(imgData.MS_LR);% low resolution MS image
        [MatrixImage_DR, MatrixResult_DR] = FusionAndEvaluateOnReduceResolution (I_GT, I_MS, I_PAN, I_MS_LR, Params);
        MatrixResults_DR = cat(3, MatrixResults_DR, MatrixResult_DR);
                
        %% 保存每组图像的融合结果
        saveName = fullfile(saveDir,[num2str(i),'.mat']);
        %{
        h1 = montage(histeq(mat2gray(MatrixImage_Fu(:,:,4:-1:2,:))),'BorderSize',10,'BackgroundColor','white');
        titleImages1 = {'PAN','EXP','PCA','IHS','Brovey','BDSD','GS','GSA','PRACS','HPF','SFIM','Indusion','ATWT','AWLP','ATWT M2','ATWT M3','MTF GLP','MTF GLP HPM PP','MTF GLP HPM','MTF GLP CBD'};
    %     saveas(h,[filename,'.jpg']);
    
    %      title('全色图像 (左)和多光谱图像 (右)');
        save(saveName,
    
        h2 = montage(histeq(mat2gray(MatrixImage_DR(:,:,4:-1:2,:))),'BorderSize',10,'BackgroundColor','white');
        titleImages2 = {'GT','EXP','PCA','IHS','Brovey','BDSD','GS','GSA','PRACS','HPF','SFIM','Indusion','ATWT','AWLP','ATWT M2','ATWT M3','MTF GLP','MTF GLP HPM PP','MTF GLP HPM','MTF GLP CBD'};
        %}
        %数据量太大，故隔十个保存一组分类结果
        %当正数与负数取余时，当得到的余数结果的符号希望跟被除数 (x)一样，用rem ()函数；当得到的余数结果的符号希望跟除数 (y)一样，用mod ()函数
%         if rem(i, 10) == 0
%             save(saveName,'loadImgPath', 'MatrixImage_Fu', 'MatrixImage_DR', 'MatrixResult_DR','MatrixResult_Fu');% 单组数据约0.6G
%         else
%             save(saveName, 'loadImgPath', 'MatrixResult_DR','MatrixResult_Fu');% 单组数据约0.6G
%         end

        % 每一个都保存
        save(saveName,'loadImgPath', 'MatrixImage_Fu', 'MatrixImage_DR', 'MatrixResult_Fu', 'MatrixResult_DR');% 单组数据约0.6G
    end
        
    % 开始统计
    MatrixResults_Fu = real(MatrixResults_Fu);
    MatrixResults_DR = real(MatrixResults_DR);
    % 计算均值
    Mean_Fu = mean(MatrixResults_Fu,3);
    Mean_DR = mean(MatrixResults_DR,3);
    %计算方差
    Var_Fu = var(MatrixResults_Fu,0,3 ); % MeanVar = [Mean_Fu,'±',Var] ;
    Var_DR = var(MatrixResults_DR,0,3 );
    
    % 计算中值 
    median_Fu = median(MatrixResults_Fu,3);
    median_DR = median(MatrixResults_DR,3);    
    % 计算最大元素和最小元素
    max_Fu = max(MatrixResults_Fu,[],3);
    max_DR = max(MatrixResults_DR,[],3);
    min_Fu = min(MatrixResults_Fu,[],3);
    min_DR = min(MatrixResults_DR,[],3);
    
    % 输出到xlsx

    % 设置xlsx文件名
    XlsxName = strcat("report", string(datetime, 'yyyy-MM-dd-HH-mm-ss'), '.xlsx');
    saveXlsxName = fullfile(saveDir,XlsxName);
    
    % Fu
    XlsxTitle = [ 'D_lambda','D_S','QNR','SAM','SCC','SD','entropy_','CEMean','SFMean', "每行一个方法,每个表格代表所有张影像在该值的统计值"]; % 指标标题                 
    writematrix(XlsxTitle,saveXlsxName,'WriteMode','append')        
    writematrix('均值Mean_Fu',saveXlsxName,'WriteMode','append');
    writematrix(Mean_Fu,saveXlsxName,'WriteMode','append');
    writematrix('方差Var_Fu',saveXlsxName,'WriteMode','append')
    writematrix(Var_Fu,saveXlsxName,'WriteMode','append')  
    

    % DR
    XlsxTitle = [ 'Q2n','Q','SAM','ERGAS','SCC_GT','RB','RV','RSD','RMSE_','QAVE_','CCMean', "每行一个方法,每个表格代表所有张影像在该值的统计值"]; % 指标标题
    writematrix(XlsxTitle,saveXlsxName,'WriteMode','append')        
    writematrix('均值Mean_DR',saveXlsxName,'WriteMode','append');
    writematrix(Mean_DR,saveXlsxName,'WriteMode','append');
    writematrix('方差Var_DR',saveXlsxName,'WriteMode','append')
    writematrix(Var_DR,saveXlsxName,'WriteMode','append')
    
    % 每个指标在不同方法中的表现统计 https://blog.csdn.net/wuli_dear_wang/article/details/88356141
    
    % Fu     for j = 1:size(MatrixResults_Fu,1)
    MatrixResults_Fu_permute = permute(MatrixResults_Fu,[1 3 2]); % (:,1,:)表示取第一列,[1 3 2]表示按矩阵变为把原来的第3维放到第2维;
    D_lambda = MatrixResults_Fu_permute(:,:,1); 
    D_S = MatrixResults_Fu_permute(:,:,2); 
    QNR = MatrixResults_Fu_permute(:,:,3); 
    SAM_MS = MatrixResults_Fu_permute(:,:,4); 
    SCC_Pan = MatrixResults_Fu_permute(:,:,5); 
    SD = MatrixResults_Fu_permute(:,:,6); 
    entropy_ = MatrixResults_Fu_permute(:,:,7); 
    CEMean = MatrixResults_Fu_permute(:,:,8); 
    SFMean = MatrixResults_Fu_permute(:,:,9); 
    saveName = fullfile(saveDir,'all_IndexFu.mat');
    save(saveName, 'D_lambda', 'D_S', 'QNR', 'SAM_MS', 'SCC_Pan', 'SD', 'entropy_', 'CEMean', 'SFMean');

    % DR
    MatrixResults_DR_permute = permute(MatrixResults_DR,[1 3 2]);
    Q2n = MatrixResults_DR_permute(:,:,1);  % (:,1,:)表示取第一列,[1 3 2]表示按矩阵变为把原来的第3维放到第2维;
    Q = MatrixResults_DR_permute(:,:,2);
    SAM = MatrixResults_DR_permute(:,:,3);
    ERGAS = MatrixResults_DR_permute(:,:,4);
    SCC_GT = MatrixResults_DR_permute(:,:,5);
    RB = MatrixResults_DR_permute(:,:,6);
    RV = MatrixResults_DR_permute(:,:,7);
    RSD = MatrixResults_DR_permute(:,:,8);
    RMSE_ = MatrixResults_DR_permute(:,:,9);
    QAVE_ = MatrixResults_DR_permute(:,:,10);
    CCMean = MatrixResults_DR_permute(:,:,11);
    saveName = fullfile(saveDir,'all_IndexDR.mat');
    save(saveName, 'Q2n','Q','SAM','ERGAS','SCC_GT','RB','RV','RSD','RMSE_','QAVE_','CCMean');

    saveName = fullfile(saveDir,'all.mat');
    formatSpec = '保存 %s mat文件！并将该二级目录统计结果打印xlsx\n ';
    fprintf(formatSpec, saveName);    
    save(saveName, 'MatrixResults_Fu', 'MatrixResults_DR','Mean_Fu','Mean_DR','Var_Fu','Var_DR','median_Fu','median_DR','max_Fu','max_DR','min_Fu','min_DR');
    
    fprintf('------------------------------\n ');
end


%%
function [MatrixImage, MatrixResults] = FusionAndEvaluateOnFullResolution (I_MS, I_MS_LR, I_PAN, Params)
%UNTITLED3 Summary of this function goes here
%   
% Inputs:
%          Fused Image;
%           I_MS_LR:            MS image;
%           I_PAN:              Panchromatic image;
%           L:                  Image radiometric resolution; 
%           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range;
%           I_MS:               MS image upsampled to the PAN size;
%           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
%           tag:                Image tag. Often equal to the field sensor. It makes sense when sensor is 'none'. It indicates the band number;
%           ratiF:              Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           D_lambda:           D_lambda index;
%           D_S:                D_S index;
%           QNR_index:          QNR index;
%           SAM_index:          Spectral Angle Mapper (SAM) index between fused and MS image;
%           sCC:                spatial Correlation Coefficient between fused and PAN images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  RUN AND FULL RESOLUTION VALIDATION  %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


cd  './Toolbox\Pansharpening Tool ver 1.3/'
%% Initialization of the Matrix of Results
% NumAlgs = 19;
% NumIndexes = 5;
% MatrixResults = zeros(NumAlgs,NumIndexes);
%% Initialization of the function parameters
% Threshold values out of dynamic range
thvalues = 0;

L = ceil(log2(double(max(I_PAN(:)))+1));% Radiometric Resolution
sensor = Params.sensor;
im_tag =  Params.sensor;
ratio = Params.ratio;
%% EXP
[D_lambda_EXP,D_S_EXP,QNRI_EXP,SAM_EXP,SCC_EXP] = indexes_evaluation_FS(I_MS,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_MS,I_MS); %
MatrixResults(1,:) = [D_lambda_EXP,D_S_EXP,QNRI_EXP,SAM_EXP,SCC_EXP,SD,entropy_,CEMean,SFMean];
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Component Substitution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PCA
%
cd PCA
t2=tic;
I_PCA = PCA(I_MS,I_PAN);
time_PCA=toc(t2);
fprintf('Elaboration time PCA: %.2f [sec]\n',time_PCA);
cd ..

[D_lambda_PCA,D_S_PCA,QNRI_PCA,SAM_PCA,SCC_PCA] = indexes_evaluation_FS(I_PCA,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_PCA,I_MS); %
MatrixResults(2,:) = [D_lambda_PCA,D_S_PCA,QNRI_PCA,SAM_PCA,SCC_PCA,SD,entropy_,CEMean,SFMean];
%% IHS

cd IHS
t2=tic;
I_IHS = IHS(I_MS,I_PAN);
time_IHS=toc(t2);
fprintf('Elaboration time IHS: %.2f [sec]\n',time_IHS);
cd ..

[D_lambda_IHS,D_S_IHS,QNRI_IHS,SAM_IHS,SCC_IHS] = indexes_evaluation_FS(I_IHS,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_IHS,I_MS); %
MatrixResults(3,:) = [D_lambda_IHS,D_S_IHS,QNRI_IHS,SAM_IHS,SCC_IHS,SD,entropy_,CEMean,SFMean];
%% Brovey

cd Brovey
t2=tic;
I_Brovey = Brovey(I_MS,I_PAN);
time_Brovey=toc(t2);
fprintf('Elaboration time Brovey: %.2f [sec]\n',time_Brovey);
cd ..

[D_lambda_Brovey,D_S_Brovey,QNRI_Brovey,SAM_Brovey,SCC_Brovey] = indexes_evaluation_FS(I_Brovey,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_Brovey,I_MS); %
MatrixResults(4,:) = [D_lambda_Brovey,D_S_Brovey,QNRI_Brovey,SAM_Brovey,SCC_Brovey,SD,entropy_,CEMean,SFMean];
%% BDSD

cd BDSD
t2=tic;

I_BDSD = BDSD(I_MS,I_PAN,ratio,128,sensor);

time_BDSD = toc(t2);
fprintf('Elaboration time BDSD: %.2f [sec]\n',time_BDSD);
cd ..

[D_lambda_BDSD,D_S_BDSD,QNRI_BDSD,SAM_BDSD,SCC_BDSD] = indexes_evaluation_FS(I_BDSD,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_BDSD,I_MS); %
MatrixResults(5,:) = [D_lambda_BDSD,D_S_BDSD,QNRI_BDSD,SAM_BDSD,SCC_BDSD,SD,entropy_,CEMean,SFMean];
%% GS

cd GS
t2=tic;
I_GS = GS(I_MS,I_PAN);
time_GS = toc(t2);
fprintf('Elaboration time GS: %.2f [sec]\n',time_GS);
cd ..

[D_lambda_GS,D_S_GS,QNRI_GS,SAM_GS,SCC_GS] = indexes_evaluation_FS(I_GS,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_GS,I_MS); %
MatrixResults(6,:) = [D_lambda_GS,D_S_GS,QNRI_GS,SAM_GS,SCC_GS,SD,entropy_,CEMean,SFMean];
%% GSA

cd GS
t2=tic;
I_GSA = GSA(I_MS,I_PAN,I_MS_LR,ratio);
time_GSA = toc(t2);
fprintf('Elaboration time GSA: %.2f [sec]\n',time_GSA);
cd ..

[D_lambda_GSA,D_S_GSA,QNRI_GSA,SAM_GSA,SCC_GSA] = indexes_evaluation_FS(I_GSA,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_GSA,I_MS); %
MatrixResults(7,:) = [D_lambda_GSA,D_S_GSA,QNRI_GSA,SAM_GSA,SCC_GSA,SD,entropy_,CEMean,SFMean];
%% PRACS

cd PRACS
t2=tic;
I_PRACS = PRACS(I_MS,I_PAN,ratio);
time_PRACS = toc(t2);
fprintf('Elaboration time PRACS: %.2f [sec]\n',time_PRACS);
cd ..

[D_lambda_PRACS,D_S_PRACS,QNRI_PRACS,SAM_PRACS,SCC_PRACS] = indexes_evaluation_FS(I_PRACS,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_PRACS,I_MS); %
MatrixResults(8,:) = [D_lambda_PRACS,D_S_PRACS,QNRI_PRACS,SAM_PRACS,SCC_PRACS,SD,entropy_,CEMean,SFMean];
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MultiResolution Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% HPF

cd HPF
t2=tic;
I_HPF = HPF(I_MS,I_PAN,ratio);
time_HPF = toc(t2);
fprintf('Elaboration time HPF: %.2f [sec]\n',time_HPF);
cd ..

[D_lambda_HPF,D_S_HPF,QNRI_HPF,SAM_HPF,SCC_HPF] = indexes_evaluation_FS(I_HPF,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_HPF,I_MS); %
MatrixResults(9,:) = [D_lambda_HPF,D_S_HPF,QNRI_HPF,SAM_HPF,SCC_HPF,SD,entropy_,CEMean,SFMean];
%% SFIM

cd SFIM
t2=tic;
I_SFIM = SFIM(I_MS,I_PAN,ratio);
time_SFIM = toc(t2);
fprintf('Elaboration time SFIM: %.2f [sec]\n',time_SFIM);
cd ..

[D_lambda_SFIM,D_S_SFIM,QNRI_SFIM,SAM_SFIM,SCC_SFIM] = indexes_evaluation_FS(I_SFIM,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_SFIM,I_MS); %
MatrixResults(10,:) = [D_lambda_SFIM,D_S_SFIM,QNRI_SFIM,SAM_SFIM,SCC_SFIM,SD,entropy_,CEMean,SFMean];
%% Indusion

cd Indusion
t2=tic;
I_Indusion = Indusion(I_PAN,I_MS_LR,ratio);
time_Indusion = toc(t2);
fprintf('Elaboration time Indusion: %.2f [sec]\n',time_Indusion);
cd ..

[D_lambda_Indusion,D_S_Indusion,QNRI_Indusion,SAM_Indusion,SCC_Indusion] = indexes_evaluation_FS(I_Indusion,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_Indusion,I_MS); %
MatrixResults(11,:) = [D_lambda_Indusion,D_S_Indusion,QNRI_Indusion,SAM_Indusion,SCC_Indusion,SD,entropy_,CEMean,SFMean];
%% ATWT

cd Wavelet
t2=tic;
I_ATWT = ATWT(I_MS,I_PAN,ratio);

time_ATWT = toc(t2);
fprintf('Elaboration time ATWT: %.2f [sec]\n',time_ATWT);
cd ..

[D_lambda_ATWT,D_S_ATWT,QNRI_ATWT,SAM_ATWT,SCC_ATWT] = indexes_evaluation_FS(I_ATWT,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_ATWT,I_MS); %
MatrixResults(12,:) = [D_lambda_ATWT,D_S_ATWT,QNRI_ATWT,SAM_ATWT,SCC_ATWT,SD,entropy_,CEMean,SFMean];
%% AWLP

cd Wavelet
t2=tic;
I_AWLP = AWLP(I_MS,I_PAN,ratio);
time_AWLP = toc(t2);
fprintf('Elaboration time AWLP: %.2f [sec]\n',time_AWLP);
cd ..

[D_lambda_AWLP,D_S_AWLP,QNRI_AWLP,SAM_AWLP,SCC_AWLP] = indexes_evaluation_FS(I_AWLP,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_AWLP,I_MS); %
MatrixResults(13,:) = [D_lambda_AWLP,D_S_AWLP,QNRI_AWLP,SAM_AWLP,SCC_AWLP,SD,entropy_,CEMean,SFMean];
%% ATWT-M2

cd Wavelet
t2=tic;

I_ATWTM2 = ATWT_M2(I_MS,I_PAN,ratio);

time_ATWTM2 = toc(t2);
fprintf('Elaboration time ATWTM2: %.2f [sec]\n',time_ATWTM2);
cd ..

[D_lambda_ATWTM2,D_S_ATWTM2,QNRI_ATWTM2,SAM_ATWTM2,SCC_ATWTM2] = indexes_evaluation_FS(I_ATWTM2,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_ATWTM2,I_MS); %
MatrixResults(14,:) = [D_lambda_ATWTM2,D_S_ATWTM2,QNRI_ATWTM2,SAM_ATWTM2,SCC_ATWTM2,SD,entropy_,CEMean,SFMean];

%% ATWT-M3

cd Wavelet
t2=tic;

I_ATWTM3 = ATWT_M3(I_MS,I_PAN,ratio); %real_SCC_ATWTM3 = real(SCC_ATWTM3);real_SFMean = real(SFMean);

time_ATWTM3 = toc(t2);
fprintf('Elaboration time ATWTM3: %.2f [sec]\n',time_ATWTM3);
cd ..

[D_lambda_ATWTM3,D_S_ATWTM3,QNRI_ATWTM3,SAM_ATWTM3,SCC_ATWTM3] = indexes_evaluation_FS(I_ATWTM3,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_ATWTM3,I_MS); %
MatrixResults(15,:) = [D_lambda_ATWTM3,D_S_ATWTM3,QNRI_ATWTM3,SAM_ATWTM3,SCC_ATWTM3,SD,entropy_,CEMean,SFMean];
%% MTF-GLP

cd GLP
t2=tic;
I_MTF_GLP = MTF_GLP(I_PAN,I_MS,sensor,im_tag,ratio);
% I_MTF_GLP = MTF_GLP_ECB(I_MS,I_PAN,ratio,[9 9],2.5,sensor,im_tag);
% I_MTF_GLP = MTF_GLP_CBD(I_MS,I_PAN,ratio,[55 55],-Inf,sensor,im_tag);

time_MTF_GLP=toc(t2);
fprintf('Elaboration time CBD: %.2f [sec]\n',time_MTF_GLP);
cd ..

[D_lambda_MTF_GLP,D_S_MTF_GLP,QNRI_MTF_GLP,SAM_MTF_GLP,SCC_MTF_GLP] = indexes_evaluation_FS(I_MTF_GLP,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_MTF_GLP,I_MS); %
MatrixResults(16,:) = [D_lambda_MTF_GLP,D_S_MTF_GLP,QNRI_MTF_GLP,SAM_MTF_GLP,SCC_MTF_GLP,SD,entropy_,CEMean,SFMean];
%% MTF-GLP-HPM-PP

cd GLP
t2=tic;
I_MTF_GLP_HPM_PP = MTF_GLP_HPM_PP(I_PAN,I_MS_LR,sensor,im_tag,ratio);
tempo_MTF_GLP_HPM_PP = toc(t2);
fprintf('Elaboration time MTF-GLP-HPM-PP: %.2f [sec]\n',tempo_MTF_GLP_HPM_PP);
cd ..

[D_lambda_MTF_GLP_HPM_PP,D_S_MTF_GLP_HPM_PP,QNRI_MTF_GLP_HPM_PP,SAM_MTF_GLP_HPM_PP,SCC_MTF_GLP_HPM_PP] = indexes_evaluation_FS(I_MTF_GLP_HPM_PP,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_MTF_GLP_HPM_PP,I_MS); %
MatrixResults(17,:) = [D_lambda_MTF_GLP_HPM_PP,D_S_MTF_GLP_HPM_PP,QNRI_MTF_GLP_HPM_PP,SAM_MTF_GLP_HPM_PP,SCC_MTF_GLP_HPM_PP,SD,entropy_,CEMean,SFMean];
%% MTF-GLP-HPM

cd GLP
t2=tic;
I_MTF_GLP_HPM = MTF_GLP_HPM(I_PAN,I_MS,sensor,im_tag,ratio);
tempo_MTF_GLP_HPM = toc(t2);
fprintf('Elaboration time MTF-GLP-HPM: %.2f [sec]\n',tempo_MTF_GLP_HPM);
cd ..

[D_lambda_MTF_GLP_HPM,D_S_MTF_GLP_HPM,QNRI_MTF_GLP_HPM,SAM_MTF_GLP_HPM,SCC_MTF_GLP_HPM] = indexes_evaluation_FS(I_MTF_GLP_HPM,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_MTF_GLP_HPM,I_MS); %
MatrixResults(18,:) = [D_lambda_MTF_GLP_HPM,D_S_MTF_GLP_HPM,QNRI_MTF_GLP_HPM,SAM_MTF_GLP_HPM,SCC_MTF_GLP_HPM,SD,entropy_,CEMean,SFMean];
%% MTF-GLP-CBD

cd GS
t2=tic;
I_MTF_GLP_CBD = GS2_GLP(I_MS,I_PAN,ratio,sensor,im_tag);
tempo_MTF_GLP_CBD = toc(t2);
fprintf('Elaboration time MTF-GLP-CBD: %.2f [sec]\n',tempo_MTF_GLP_CBD);
cd ..

[D_lambda_MTF_GLP_CBD,D_S_MTF_GLP_CBD,QNRI_MTF_GLP_CBD,SAM_MTF_GLP_CBD,SCC_MTF_GLP_CBD] = indexes_evaluation_FS(I_MTF_GLP_CBD,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
[SD,entropy_,CEMean,SFMean] = FusionImg2EvaluationMetricFu(I_MTF_GLP_CBD,I_MS); %
MatrixResults(19,:) = [D_lambda_MTF_GLP_CBD,D_S_MTF_GLP_CBD,QNRI_MTF_GLP_CBD,SAM_MTF_GLP_CBD,SCC_MTF_GLP_CBD,SD,entropy_,CEMean,SFMean];

%% Print in LATEX
% matrix2latex(MatrixResults,'Real_Dataset.tex', 'rowLabels',[{'EXP'},{'PCA'},{'IHS'},{'Brovey'},{'BDSD'},{'GS'},{'GSA'},{'PRACS'},{'HPF'},{'SFIM'},{'Indusion'},{'ATWT'},{'AWLP'},...
%         {'ATWT-M2'},{'ATWT-M3'},{'MTF-GLP'},{'MTF-GLP-HPM-PP'},{'MTF-GLP-HPM'},{'MTF-GLP-CBD'}],'columnLabels',[{'D_{\lambda}'},{'D_{S}'},{'QNR'}],'alignment','c','format', '%.4f'); 
    % matrix2latex(MatrixResults,'Real_Dataset.tex', 'rowLabels',[{'EXP'},{'PCA'},{'IHS'},{'Brovey'},{'BDSD'},{'GS'},{'GSA'},{'PRACS'},{'HPF'},{'SFIM'},{'Indusion'},{'ATWT'},{'AWLP'},...
    %     {'ATWT-M2'},{'ATWT-M3'},{'MTF-GLP'},{'MTF-GLP-HPM-PP'},{'MTF-GLP-HPM'},{'MTF-GLP-CBD'}],'columnLabels',[{'D_{\lambda}'},{'D_{S}'},{'QNR'},{'SAM'},{'SCC'}],'alignment','c','format', '%.4f'); 
%% View All
MatrixImage(:,:,:,1) = repmat(I_PAN,[1 1 size(I_MS,3)]);
MatrixImage(:,:,:,2) = I_MS;
MatrixImage(:,:,:,3) = I_PCA;
MatrixImage(:,:,:,4) = I_IHS;
MatrixImage(:,:,:,5) = I_Brovey;
MatrixImage(:,:,:,6) = I_BDSD;
MatrixImage(:,:,:,7) = I_GS;
MatrixImage(:,:,:,8) = I_GSA;
MatrixImage(:,:,:,9) = I_PRACS;
MatrixImage(:,:,:,10) = I_HPF;
MatrixImage(:,:,:,11) = I_SFIM;
MatrixImage(:,:,:,12) = I_Indusion;
MatrixImage(:,:,:,13) = I_ATWT;
MatrixImage(:,:,:,14) = I_AWLP;
MatrixImage(:,:,:,15) = I_ATWTM2;
MatrixImage(:,:,:,16) = I_ATWTM3;
MatrixImage(:,:,:,17) = I_MTF_GLP;
MatrixImage(:,:,:,18) = I_MTF_GLP_HPM_PP;
MatrixImage(:,:,:,19) = I_MTF_GLP_HPM;
MatrixImage(:,:,:,20) = I_MTF_GLP_CBD;
%% 显示
%{
if size(I_MS,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,1];
end

flag_cut_bounds = 0;
dim_cut = 0;
titleImages = {'PAN','EXP','PCA','IHS','Brovey','BDSD','GS','GSA','PRACS','HPF','SFIM','Indusion','ATWT','AWLP','ATWT M2','ATWT M3','MTF GLP','MTF GLP HPM PP','MTF GLP HPM','MTF GLP CBD'};
figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,1);
%}
cd ../../ %返回主文件夹
end

%%
function [MatrixImage, MatrixResults] = FusionAndEvaluateOnReduceResolution (I_GT, I_MS, I_PAN,I_MS_LR, Params)
%UNTITLED3 Summary of this function goes here
%   
% Inputs:
%             I_GT:      grouth truth 
%             I_MS:     MS upsample to pan size    
%             I_PAN    orginial Pan 
%             I_MS_LR  orginial MS image 
%           ratiF:              Scale ratio between MS and PAN. Pre-condition: Integer value;
%           L:                  Image radiometric resolution; 
%           Q_blocks_size:      Block size of the Q-index locally applied;
%           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
%           dim_cut:            Define the dimension of the boundary cut;
%           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range.
%
% Outputs:
%           Q_index:            Q index;
%           SAM_index:          Spectral Angle Mapper (SAM) index;
%           ERGAS_index:        Erreur Relative Globale Adimensionnelle de Synth鑣e (ERGAS) index;
%           sCC:                spatial Correlation Coefficient between fused and ground-truth images;
%           Q2n_index:          Q2n index.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  RUN AND FULL RESOLUTION VALIDATION  %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


cd  './Toolbox\Pansharpening Tool ver 1.3/'
%% Initialization of the Matrix of Results
% NumAlgs = 19;
% NumIndexes = 5;
% MatrixResults = zeros(NumAlgs,NumIndexes);
%% Initialization of the function parameters
% Threshold values out of dynamic range
ratio = Params.ratio;
L = ceil(log2(double(max(I_PAN(:)))+1));% Radiometric Resolution
Qblocks_size = 32;

flag_cut_bounds = 0;%不进行裁切
dim_cut = 0;%裁剪的大小不设置
thvalues = 0;

sensor = Params.sensor;
im_tag =  Params.sensor;
%% GT
[Q_avg_GT, SAM_GT, ERGAS_GT, SCC_GT, Q_GT] = indexes_evaluation(I_GT,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%% EXP
[Q_avg_EXP, SAM_EXP, ERGAS_EXP, SCC_GT_EXP, Q_EXP] = indexes_evaluation(I_MS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_MS,I_GT);
MatrixResults(1,:) = [Q_EXP,Q_avg_EXP,SAM_EXP,ERGAS_EXP,SCC_GT_EXP,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Component Substitution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PCA
cd PCA
t2=tic;
I_PCA = PCA(I_MS,I_PAN);
time_PCA=toc(t2);
fprintf('Elaboration time PCA: %.2f [sec]\n',time_PCA);
cd ..

[Q_avg_PCA, SAM_PCA, ERGAS_PCA, SCC_GT_PCA, Q_PCA] = indexes_evaluation(I_PCA,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_PCA,I_GT);
MatrixResults(2,:) = [Q_PCA,Q_avg_PCA,SAM_PCA,ERGAS_PCA,SCC_GT_PCA,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% IHS

cd IHS
t2=tic;
I_IHS = IHS(I_MS,I_PAN);
time_IHS=toc(t2);
fprintf('Elaboration time IHS: %.2f [sec]\n',time_IHS);
cd ..

[Q_avg_IHS, SAM_IHS, ERGAS_IHS, SCC_GT_IHS, Q_IHS] = indexes_evaluation(I_IHS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_IHS,I_GT);
MatrixResults(3,:) = [Q_IHS,Q_avg_IHS,SAM_IHS,ERGAS_IHS,SCC_GT_IHS,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% Brovey
cd Brovey
t2=tic;
I_Brovey = Brovey(I_MS,I_PAN);
time_Brovey=toc(t2);
fprintf('Elaboration time Brovey: %.2f [sec]\n',time_Brovey);
cd ..

[Q_avg_Brovey, SAM_Brovey, ERGAS_Brovey, SCC_GT_Brovey, Q_Brovey] = indexes_evaluation(I_Brovey,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_Brovey,I_GT);
MatrixResults(4,:) = [Q_Brovey,Q_avg_Brovey,SAM_Brovey,ERGAS_Brovey,SCC_GT_Brovey,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% BDSD

cd BDSD
t2=tic;

I_BDSD = BDSD(I_MS,I_PAN,ratio,size(I_MS,1),sensor);

time_BDSD = toc(t2);
fprintf('Elaboration time BDSD: %.2f [sec]\n',time_BDSD);
cd ..

[Q_avg_BDSD, SAM_BDSD, ERGAS_BDSD, SCC_GT_BDSD, Q_BDSD] = indexes_evaluation(I_BDSD,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_BDSD,I_GT);
MatrixResults(5,:) = [Q_BDSD,Q_avg_BDSD,SAM_BDSD,ERGAS_BDSD,SCC_GT_BDSD,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% GS

cd GS
t2=tic;
I_GS = GS(I_MS,I_PAN);
time_GS = toc(t2);
fprintf('Elaboration time GS: %.2f [sec]\n',time_GS);
cd ..

[Q_avg_GS, SAM_GS, ERGAS_GS, SCC_GT_GS, Q_GS] = indexes_evaluation(I_GS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_GS,I_GT);
MatrixResults(6,:) = [Q_GS,Q_avg_GS,SAM_GS,ERGAS_GS,SCC_GT_GS,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% GSA

cd GS
t2=tic;
I_GSA = GSA(I_MS,I_PAN,I_MS_LR,ratio);
tempo_GSA = toc(t2);
fprintf('Elaboration time GSA: %.2f [sec]\n',tempo_GSA);
cd ..

[Q_avg_GSA, SAM_GSA, ERGAS_GSA, SCC_GT_GSA, Q_GSA] = indexes_evaluation(I_GSA,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_GSA,I_GT);
MatrixResults(7,:) = [Q_GSA,Q_avg_GSA,SAM_GSA,ERGAS_GSA,SCC_GT_GSA,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% PRACS

cd PRACS
t2=tic;
I_PRACS = PRACS(I_MS,I_PAN,ratio);
time_PRACS = toc(t2);
fprintf('Elaboration time PRACS: %.2f [sec]\n',time_PRACS);
cd ..

[Q_avg_PRACS, SAM_PRACS, ERGAS_PRACS, SCC_GT_PRACS, Q_PRACS] = indexes_evaluation(I_PRACS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_PRACS,I_GT);
MatrixResults(8,:) = [Q_PRACS,Q_avg_PRACS,SAM_PRACS,ERGAS_PRACS,SCC_GT_PRACS,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MultiResolution Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% HPF
cd HPF
t2=tic;
I_HPF = HPF(I_MS,I_PAN,ratio);
time_HPF = toc(t2);
fprintf('Elaboration time HPF: %.2f [sec]\n',time_HPF);
cd ..

[Q_avg_HPF, SAM_HPF, ERGAS_HPF, SCC_GT_HPF, Q_HPF] = indexes_evaluation(I_HPF,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_HPF,I_GT);
MatrixResults(9,:) = [Q_HPF,Q_avg_HPF,SAM_HPF,ERGAS_HPF,SCC_GT_HPF,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% SFIM

cd SFIM
t2=tic;
I_SFIM = SFIM(I_MS,I_PAN,ratio);
time_SFIM = toc(t2);
fprintf('Elaboration time SFIM: %.2f [sec]\n',time_SFIM);
cd ..

[Q_avg_SFIM, SAM_SFIM, ERGAS_SFIM, SCC_GT_SFIM, Q_SFIM] = indexes_evaluation(I_SFIM,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_SFIM,I_GT);
MatrixResults(10,:) = [Q_SFIM,Q_avg_SFIM,SAM_SFIM,ERGAS_SFIM,SCC_GT_SFIM,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% Indusion

cd Indusion
t2=tic;
I_Indusion = Indusion(I_PAN,I_MS_LR,ratio);
time_Indusion = toc(t2);
fprintf('Elaboration time Indusion: %.2f [sec]\n',time_Indusion);
cd ..

[Q_avg_Indusion, SAM_Indusion, ERGAS_Indusion, SCC_GT_Indusion, Q_Indusion] = indexes_evaluation(I_Indusion,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_Indusion,I_GT);
MatrixResults(11,:) = [Q_Indusion,Q_avg_Indusion,SAM_Indusion,ERGAS_Indusion,SCC_GT_Indusion,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% ATWT

cd Wavelet
t2=tic;
I_ATWT = ATWT(I_MS,I_PAN,ratio);
time_ATWT = toc(t2);
fprintf('Elaboration time ATWT: %.2f [sec]\n',time_ATWT);
cd ..

[Q_avg_ATWT, SAM_ATWT, ERGAS_ATWT, SCC_GT_ATWT, Q_ATWT] = indexes_evaluation(I_ATWT,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_ATWT,I_GT);
MatrixResults(12,:) = [Q_ATWT,Q_avg_ATWT,SAM_ATWT,ERGAS_ATWT,SCC_GT_ATWT,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% AWLP

cd Wavelet
t2=tic;
I_AWLP = AWLP(I_MS,I_PAN,ratio);
time_AWLP = toc(t2);
fprintf('Elaboration time AWLP: %.2f [sec]\n',time_AWLP);
cd ..

[Q_avg_AWLP, SAM_AWLP, ERGAS_AWLP, SCC_GT_AWLP, Q_AWLP] = indexes_evaluation(I_AWLP,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_AWLP,I_GT);
MatrixResults(13,:) = [Q_AWLP,Q_avg_AWLP,SAM_AWLP,ERGAS_AWLP,SCC_GT_AWLP,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% ATWT-M2

cd Wavelet
t2=tic;

I_ATWTM2 = ATWT_M2(I_MS,I_PAN,ratio);

time_ATWTM2 = toc(t2);
fprintf('Elaboration time ATWT-M2: %.2f [sec]\n',time_ATWTM2);
cd ..

[Q_avg_ATWTM2, SAM_ATWTM2, ERGAS_ATWTM2, SCC_GT_ATWTM2, Q_ATWTM2] = indexes_evaluation(I_ATWTM2,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_ATWTM2,I_GT);
MatrixResults(14,:) = [Q_ATWTM2,Q_avg_ATWTM2,SAM_ATWTM2,ERGAS_ATWTM2,SCC_GT_ATWTM2,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% ATWT-M3

cd Wavelet
t2=tic;

I_ATWTM3 = ATWT_M3(I_MS,I_PAN,ratio);

time_ATWTM3 = toc(t2);
fprintf('Elaboration time ATWT-M3: %.2f [sec]\n',time_ATWTM3);
cd ..

[Q_avg_ATWTM3, SAM_ATWTM3, ERGAS_ATWTM3, SCC_GT_ATWTM3, Q_ATWTM3] = indexes_evaluation(I_ATWTM3,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_ATWTM3,I_GT);
MatrixResults(15,:) = [Q_ATWTM3,Q_avg_ATWTM3,SAM_ATWTM3,ERGAS_ATWTM3,SCC_GT_ATWTM3,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% MTF-GLP
cd GLP
t2=tic;
I_MTF_GLP = MTF_GLP(I_PAN,I_MS,sensor,im_tag,ratio);
% I_MTF_GLP = MTF_GLP_ECB(I_MS,I_PAN,ratio,[9 9],2.5,sensor,im_tag);
% I_MTF_GLP = MTF_GLP_CBD(I_MS,I_PAN,ratio,[55 55],-Inf,sensor,im_tag);
time_MTF_GLP = toc(t2);
fprintf('Elaboration time MTF-GLP: %.2f [sec]\n',time_MTF_GLP);
cd ..

[Q_avg_MTF_GLP, SAM_MTF_GLP, ERGAS_MTF_GLP, SCC_GT_MTF_GLP, Q_MTF_GLP] = indexes_evaluation(I_MTF_GLP,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_MTF_GLP,I_GT);
MatrixResults(16,:) = [Q_MTF_GLP,Q_avg_MTF_GLP,SAM_MTF_GLP,ERGAS_MTF_GLP,SCC_GT_MTF_GLP,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% MTF-GLP-HPM-PP

cd GLP
t2=tic;
I_MTF_GLP_HPM_PP = MTF_GLP_HPM_PP(I_PAN,I_MS_LR,sensor,im_tag,ratio);
time_MTF_GLP_HPM_PP = toc(t2);
fprintf('Elaboration time MTF-GLP-HPM-PP: %.2f [sec]\n',time_MTF_GLP_HPM_PP);
cd ..

[Q_avg_MTF_GLP_HPM_PP, SAM_MTF_GLP_HPM_PP, ERGAS_MTF_GLP_HPM_PP, SCC_GT_MTF_GLP_HPM_PP, Q_MTF_GLP_HPM_PP] = indexes_evaluation(I_MTF_GLP_HPM_PP,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_MTF_GLP_HPM_PP,I_GT);
MatrixResults(17,:) = [Q_MTF_GLP_HPM_PP,Q_avg_MTF_GLP_HPM_PP,SAM_MTF_GLP_HPM_PP,ERGAS_MTF_GLP_HPM_PP,SCC_GT_MTF_GLP_HPM_PP,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% MTF-GLP-HPM

cd GLP
t2=tic;
I_MTF_GLP_HPM = MTF_GLP_HPM(I_PAN,I_MS,sensor,im_tag,ratio);
time_MTF_GLP_HPM = toc(t2);
fprintf('Elaboration time MTF-GLP-HPM: %.2f [sec]\n',time_MTF_GLP_HPM);
cd ..

[Q_avg_MTF_GLP_HPM, SAM_MTF_GLP_HPM, ERGAS_MTF_GLP_HPM, SCC_GT_MTF_GLP_HPM, Q_MTF_GLP_HPM] = indexes_evaluation(I_MTF_GLP_HPM,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_MTF_GLP_HPM,I_GT);
MatrixResults(18,:) = [Q_MTF_GLP_HPM,Q_avg_MTF_GLP_HPM,SAM_MTF_GLP_HPM,ERGAS_MTF_GLP_HPM,SCC_GT_MTF_GLP_HPM,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% MTF-GLP-CBD

cd GS
t2=tic;

I_MTF_GLP_CBD = GS2_GLP(I_MS,I_PAN,ratio,sensor,im_tag);

time_MTF_GLP_CBD = toc(t2);
fprintf('Elaboration time MTF-GLP-CBD: %.2f [sec]\n',time_MTF_GLP_CBD);
cd ..

[Q_avg_MTF_GLP_CBD, SAM_MTF_GLP_CBD, ERGAS_MTF_GLP_CBD, SCC_GT_MTF_GLP_CBD, Q_MTF_GLP_CBD] = indexes_evaluation(I_MTF_GLP_CBD,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
[RB,RV,RSD,RMSE_,QAVE_,CCMean] = FusionImg2EvaluationMetricDR(I_MTF_GLP_CBD,I_GT);
MatrixResults(19,:) = [Q_MTF_GLP_CBD,Q_avg_MTF_GLP_CBD,SAM_MTF_GLP_CBD,ERGAS_MTF_GLP_CBD,SCC_GT_MTF_GLP_CBD,RB,RV,RSD,RMSE_,QAVE_,CCMean];
%% Print in LATEX
% 
% if size(I_GT,3) == 4
%    matrix2latex(MatrixResults,'Real_Dataset.tex', 'rowLabels',[{'EXP'},{'PCA'},{'IHS'},{'Brovey'},{'BDSD'},{'GS'},{'GSA'},{'PRACS'},{'HPF'},{'SFIM'},{'Indusion'},{'ATWT'},{'AWLP'},...
%         {'ATWT-M2'},{'ATWT-M3'},{'MTF-GLP'},{'MTF-GLP-HPM-PP'},{'MTF-GLP-HPM'},{'MTF-GLP-CBD'}],'columnLabels',[{'Q4'},{'Q'},{'SAM'},{'ERGAS'},{'SCC'}],'alignment','c','format', '%.4f');
% else
%    matrix2latex(MatrixResults,'Real_Dataset.tex', 'rowLabels',[{'EXP'},{'PCA'},{'IHS'},{'Brovey'},{'BDSD'},{'GS'},{'GSA'},{'PRACS'},{'HPF'},{'SFIM'},{'Indusion'},{'ATWT'},{'AWLP'},...
%         {'ATWT-M2'},{'ATWT-M3'},{'MTF-GLP'},{'MTF-GLP-HPM-PP'},{'MTF-GLP-HPM'},{'MTF-GLP-CBD'}],'columnLabels',[{'Q8'},{'Q'},{'SAM'},{'ERGAS'},{'SCC'}],'alignment','c','format', '%.4f'); 
% end

%% View All
MatrixImage(:,:,:,1) = I_GT;
MatrixImage(:,:,:,2) = I_MS;
MatrixImage(:,:,:,3) = I_PCA;
MatrixImage(:,:,:,4) = I_IHS;
MatrixImage(:,:,:,5) = I_Brovey;
MatrixImage(:,:,:,6) = I_BDSD;
MatrixImage(:,:,:,7) = I_GS;
MatrixImage(:,:,:,8) = I_GSA;
MatrixImage(:,:,:,9) = I_PRACS;
MatrixImage(:,:,:,10) = I_HPF;
MatrixImage(:,:,:,11) = I_SFIM;
MatrixImage(:,:,:,12) = I_Indusion;
MatrixImage(:,:,:,13) = I_ATWT;
MatrixImage(:,:,:,14) = I_AWLP;
MatrixImage(:,:,:,15) = I_ATWTM2;
MatrixImage(:,:,:,16) = I_ATWTM3;
MatrixImage(:,:,:,17) = I_MTF_GLP;
MatrixImage(:,:,:,18) = I_MTF_GLP_HPM_PP;
MatrixImage(:,:,:,19) = I_MTF_GLP_HPM;
MatrixImage(:,:,:,20) = I_MTF_GLP_CBD;
%%
%{
if size(I_GT,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,1];
end

titleImages = {'GT','EXP','PCA','IHS','Brovey','BDSD','GS','GSA','PRACS','HPF','SFIM','Indusion','ATWT','AWLP','ATWT M2','ATWT M3','MTF GLP','MTF GLP HPM PP','MTF GLP HPM','MTF GLP CBD'};
flag_cut_bounds = 0;
dim_cut = 0;
figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);
%}
cd ../../ %返回主文件夹
end

%%
% 附录：
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo_FusionComparsions.m
% 
% 采用benchmark数据集进行融合实验，在高低两个分辨率尺度上对融合结果进行评价
% 对比方法采用 Pansharpening Tool ver 1.3中的方法；
% 对比指标主要是
% 
% 
% clc
% clear
% close all
% % addpath(genpath('./Toolbox\'));
% %%
% %全部Fusion数据所在路径
% % ImgPaths = '.\Benchmark_Output\GF1\1\';
% % ImgPaths = '..\fenleiRS\GF1\Js\';
% ImgPaths = '.\temp\GF1\Js\';
% %数据保存路径
% % saveDir = '.\FusionComparsionsResults\GF1\1\';%设置对应保存路径
% saveDir = '.\temp\GF1\JsResults\';%设置对应保存路径
% 
% if ~exist(saveDir,'dir')%待保存的图像文件夹不存在，就建文件夹
%     mkdir(saveDir)
% end
% %%
% 
% %列出传感器文件夹内所有的融合
% listing = dir([ImgPaths,'**/*.mat']) ;
% NumImgs = size(listing,1);
% MatrixResults_Fu = zeros(19, 5, NumImgs);%存储融合结果的矩阵
% MatrixResults_DR = zeros(19, 5, NumImgs);%存储j降分辨率融合结果的矩阵
% % parfor i = 1:NumImgs
%  for i = 1:NumImgs
% 
%     
%     formatSpec = '处理%d个图像中第%d个！\n';
%     fprintf(formatSpec, NumImgs, i);
%     
%     loadImgPath = [listing(i).folder,'\',listing(i).name];
%     imgData = load(loadImgPath);
%     
%     %Full resoution results
%     I_MS_LR = double(imgData.MS); % MS image;
%     I_MS =  double(imgData.MS_Up);% MS image upsampled to the PAN size;
%     I_PAN = double(imgData.Pan); %Pan
%     Params = imgData.Paras;
%     [MatrixImage_Fu, MatrixResult_Fu] = FusionAndEvaluateOnFullResolution (I_MS, I_MS_LR, I_PAN, Params);
%     
%     %Reduced resoution results
%     I_GT = double(imgData.MS); %ground truth
%     I_PAN = double(imgData.Pan_LR);% low resolution Pan image
%     I_MS  = double(imgData.MS_LR_Up);% low resolution MS image upsampled at  low resolution  PAN scale;
%     I_MS_LR = double(imgData.MS_LR);% low resolution MS image
%     [MatrixImage_DR, MatrixResult_DR] = FusionAndEvaluateOnReduceResolution (I_GT, I_MS, I_PAN, I_MS_LR, Params);
%     
%     MatrixResults_DR(:,:,i) = MatrixResult_DR;
%     MatrixResults_Fu(:,:,i)= MatrixResult_Fu;
%     %% 保存每组图像的融合结果
%     saveName = fullfile(saveDir,[num2str(i),'.mat']);
%     %{
%     h1 = montage(histeq(mat2gray(MatrixImage_Fu(:,:,4:-1:2,:))),'BorderSize',10,'BackgroundColor','white');
%     titleImages1 = {'PAN','EXP','PCA','IHS','Brovey','BDSD','GS','GSA','PRACS','HPF','SFIM','Indusion','ATWT','AWLP','ATWT M2','ATWT M3','MTF GLP','MTF GLP HPM PP','MTF GLP HPM','MTF GLP CBD'};
% %     saveas(h,[filename,'.jpg']);
% 
% %      title('全色图像 (左)和多光谱图像 (右)');
%     save(saveName,
% 
%     h2 = montage(histeq(mat2gray(MatrixImage_DR(:,:,4:-1:2,:))),'BorderSize',10,'BackgroundColor','white');
%     titleImages2 = {'GT','EXP','PCA','IHS','Brovey','BDSD','GS','GSA','PRACS','HPF','SFIM','Indusion','ATWT','AWLP','ATWT M2','ATWT M3','MTF GLP','MTF GLP HPM PP','MTF GLP HPM','MTF GLP CBD'};
%     %}
%     %数据量太大，故隔十个保存一组分类结果
%     %当正数与负数取余时，当得到的余数结果的符号希望跟被除数 (x)一样，用rem ()函数；当得到的余数结果的符号希望跟除数 (y)一样，用mod ()函数
%     if rem(i, 10) == 0
%         save(saveName,'MatrixImage_Fu', 'MatrixImage_DR', 'loadImgPath', 'MatrixResult_DR','MatrixResult_Fu');% 单组数据约0.6G
%     else
%         save(saveName, 'loadImgPath', 'MatrixResult_DR','MatrixResult_Fu');% 单组数据约0.6G
%     end
% end
% saveName = fullfile(saveDir,'all.mat');
% save(saveName, 'MatrixResults_Fu', 'MatrixImage_DR','ImgPaths');