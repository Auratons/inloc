function parfor_denseGV(qCell, allDBCell, dbPath, params)
    qname = qCell.img_name
    coarselayerlevel = get_with_default(params, 'input_feature_layer', 5);
    finelayerlevel = 3;

    this_densegv_matname = fullfile(params.output_gv_dense_dir, filename(qname), strcat(filename(dbPath), ".mat"));
    if exist(this_densegv_matname, 'file') ~= 2

        % load input feature
        idx = find(strcmp(convertCharsToStrings(dbPath), {allDBCell.img_path}));
        cnndb = db_features(idx).features;
        % cnndb = cnndb.cnn;

        % coarse-to-fine matching
        cnnfeat1size = size(qCell.features{finelayerlevel}.x);
        cnnfeat2size = size(cnndb{finelayerlevel}.x);
        [match12, f1, f2, cnnfeat1, cnnfeat2] = at_coarse2fine_matching(qCell.features, cnndb, coarselayerlevel, finelayerlevel);
        [inls12] = at_denseransac(f1, f2, match12, 2);

        % Assert parent dir is pre-created for thread-safety. This is called from parfor.
        save('-v6', this_densegv_matname, 'cnnfeat1size', 'cnnfeat2size', 'f1', 'f2', 'inls12', 'match12');


    %     %debug
    %     im1 = imresize(imread(fullfile(params.dataset.query.dir, qname)), cnnfeat1size(1:2));
    %     im2 = imresize(imread(fullfile(params.dataset.db.cutouts.dir, dbPath)), cnnfeat2size(1:2));
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
    [~, name, ext] = fileparts(pth);
    name = [name, ext];
end
