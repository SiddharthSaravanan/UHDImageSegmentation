% Demo for Structured Edge Detector (please see readme.txt first).
function E = edgesEval(img_fname)

    %% set opts for training (see edgesTrain.m)
    opts=edgesTrain();                % default options (good settings)
    opts.modelDir='models/';          % model will be in models/forest
    opts.modelFnm='modelBsds';        % model name
    opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
    opts.useParfor=0;                 % parallelize if sufficient memory

    %% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
    model=edgesTrain(opts); % will load model if already trained

    %% set detection parameters (can set after training)
    model.opts.multiscale=1;          % for top accuracy set multiscale=1
    model.opts.sharpen=2;             % for top speed set sharpen=0
    model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
    model.opts.nThreads=4;            % max number threads for evaluation
    model.opts.nms=0;                 % set to true to enable nms

    %% evaluate edge detector on BSDS500 (see edgesEval.m)
    % if(0), edgesEval( model, 'show',1, 'name','' ); end

    %% detect edge and visualize results
    I = imread(img_fname);
    E=edgesDetect(I,model);
end