function parfor_denseGV( cnnq, qname, dbname, params )
    [filepath, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'at_netvlad_function'));
    addpath(fullfile(filepath, '..', 'yael_matlab_linux64_v438'));
    addpath(fullfile(filepath, '..', 'inLocCIIRC_utils'));

    coarselayerlevel = get_with_default(params, 'input_feature_layer', 6);
    finelayerlevel = 3;

    this_densegv_matname = fullfile(params.output_gv_dense_dir, filename(qname), strcat(filename(dbname), ".mat"));
    load(params.input_db_features_mat_path, 'features');

    if exist(this_densegv_matname, 'file') ~= 2

        % vload input feature
        idx = find(strcmp(convertCharsToStrings(dbname), {features.img_path}));
        cnndb = features(idx).features;
        % cnndb = cnndb.cnn;

        % coarse-to-fine matching
        cnnfeat1size = size(cnnq{finelayerlevel}.x);
        cnnfeat2size = size(cnndb{finelayerlevel}.x);
        [match12,f1,f2,cnnfeat1,cnnfeat2] = at_coarse2fine_matching(cnnq,cnndb,coarselayerlevel,finelayerlevel);
        [inls12] = at_denseransac(f1,f2,match12,2);

        save('-v6', this_densegv_matname, 'cnnfeat1size', 'cnnfeat2size', 'f1', 'f2', 'inls12', 'match12');


    %     %debug
    %     im1 = imresize(imread(fullfile(params.dataset.query.dir, qname)), cnnfeat1size(1:2));
    %     im2 = imresize(imread(fullfile(params.dataset.db.cutouts.dir, dbname)), cnnfeat2size(1:2));
    %     figure();
    %     ultimateSubplot ( 2, 1, 1, 1, 0.01, 0.05 );
    %     imshow(rgb2gray(im1));hold on;
    %     plot(f1(1,match12(1,:)),f1(2,match12(1,:)),'b.');
    %     plot(f1(1,inls12(1,:)),f1(2,inls12(1,:)),'g.');
    %     ultimateSubplot ( 2, 1, 2, 1, 0.01, 0.05 );
    %     imshow(rgb2gray(im2));hold on;
    %     plot(f2(1,match12(2,:)),f2(2,match12(2,:)),'b.');
    %     plot(f2(1,inls12(2,:)),f2(2,inls12(2,:)),'g.');
    %     keyboard;
    end
end

function name = filename(pth)
    splits = strsplit(pth, filesep);
    name = splits(length(splits));
    name = name{:};
end
