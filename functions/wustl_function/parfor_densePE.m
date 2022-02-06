function parfor_densePE( qname, dbname, params )
    [filepath, ~, ~] = fileparts(mfilename('fullpath'));
    addpath(fullfile(filepath, '..', 'at_netvlad_function'));
    addpath(fullfile(filepath, '..', 'ht_pnp_function'));
    % addpath(absFnPath('npy-matlab', 'npy-matlab'));
    % addpath(absFnPath('colmap', 'scripts', 'matlab'));

    [~, dbbasename] = fileparts(filename(dbname));
    this_densepe_matname = fullfile(params.output_pnp_dense_inlier_dir, filename(qname), strcat(dbbasename, ".mat"));

    if exist(this_densepe_matname, 'file') ~= 2
        %geometric verification results
        this_densegv_matname = fullfile(params.output_gv_dense_dir, filename(qname), strcat(filename(dbname), ".mat"));
        this_gvresults = load(this_densegv_matname);
        tent_xq2d = this_gvresults.f1(:, this_gvresults.inls12(1, :));
        tent_xdb2d = this_gvresults.f2(:, this_gvresults.inls12(2, :));

        %depth information
        this_db_matname = fullfile(params.input_cutout_matfiles_path, strcat("cutout_", erase(filename(dbname), "_reference"), ".mat"));
        load(this_db_matname, 'XYZcut');
        %load transformation matrix (local to global)
        transformation_txtname = fullfile(params.input_transforms_path, strcat("cutout_", erase(filename(dbname), "_reference"), ".mat"));
        load(transformation_txtname, 'R', 'position', 'calibration_mat');
        P_after = eye(4);
        P_after(1:3, 1:3) = R;
        P_after(1:3, 4) = position;
        %[ ~, P_after ] = load_WUSTL_transformation(transformation_txtname);
        %Feature upsampling
        Iqsize = size(imread(qname));
        Idbsize = size(XYZcut);
        tent_xq2d = at_featureupsample(tent_xq2d,this_gvresults.cnnfeat1size,Iqsize);
        tent_xdb2d = at_featureupsample(tent_xdb2d,this_gvresults.cnnfeat2size,Idbsize);
        %query ray
        tent_ray2d = calibration_mat^-1 * [tent_xq2d; ones(1, size(tent_xq2d, 2))];
        %DB 3d points
        indx = sub2ind(size(XYZcut(:,:,1)),tent_xdb2d(2,:),tent_xdb2d(1,:));
        X = XYZcut(:,:,1);Y = XYZcut(:,:,2);Z = XYZcut(:,:,3);
        tent_xdb3d = [X(indx); Y(indx); Z(indx)];
        tent_xdb3d = bsxfun(@plus, P_after(1:3, 1:3)*tent_xdb3d, P_after(1:3, 4));
        %Select keypoint correspond to 3D
        idx_3d = all(~isnan(tent_xdb3d), 1);
        tent_xq2d = tent_xq2d(:, idx_3d);
        tent_xdb2d = tent_xdb2d(:, idx_3d);
        tent_ray2d = tent_ray2d(:, idx_3d);
        tent_xdb3d = tent_xdb3d(:, idx_3d);


        tentatives_2d = [tent_xq2d; tent_xdb2d];
        tentatives_3d = [tent_ray2d; tent_xdb3d];


        %solver
        if size(tentatives_2d, 2) < 3
            P = nan(3, 4);
            inls = false(1, size(tentatives_2d, 2));
        else
            [ P, inls ] = ht_lo_ransac_p3p( tent_ray2d, tent_xdb3d, 1.0*pi/180);
            if isempty(P)
                P = nan(3, 4);
            end
        end

        save('-v6', this_densepe_matname, 'P', 'inls', 'tentatives_2d', 'tentatives_3d');


    end

end


% function parfor_densePE( qname, dbname, dbdepthname, params )
%
%     addpath(absFnPath('npy-matlab', 'npy-matlab'));
%     addpath(absFnPath('colmap', 'scripts', 'matlab'));
%
%     [~, dbbasename] = fileparts(filename(dbname));
%     this_densepe_matname = fullfile(params.output_pnp_dense_inlier_dir, filename(qname), strcat(dbbasename, ".mat"));
%
%     if exist(this_densepe_matname, 'file') ~= 2
%         % geometric verification results
%         this_densegv_matname = fullfile(params.output_gv_dense_dir, filename(qname), strcat(filename(dbname), ".mat"));
%         this_gvresults = load(this_densegv_matname);
%         tent_xq2d = this_gvresults.f1(:, this_gvresults.inls12(1, :));
%         tent_xdb2d = this_gvresults.f2(:, this_gvresults.inls12(2, :));
%
%         % depth information
%         depth = readNPY(dbdepthname);
%         % load transformation matrix (local to global)
%         [cameras, images] = read_model(params.input_transforms_path);
%         split = strsplit(filename(dbname), '_');
%         db_img_name = str2num(split{1});
%         cam_params = cameras(db_img_name).params;
%         r = images(db_img_name).R;
%         t = images(db_img_name).t;
%         P_after = eye(4);
%         P_after(1:3, 1:3) = r';
%         P_after(1:3, 4) = -r' * t;
%         P_after(1:4, 1:3) = P_after(1:4, 1:3) * -1;
%
%         if any(strcmp(cameras(str2num(split{1})).model, ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]))
%             Kq = [cam_params(1), 0,             cam_params(2); ...
%                   0,             cam_params(1), cam_params(3); ...
%                   0,             0,             1];
%         else  % "RADIAL", "PINHOLE"
%             Kq = [cam_params(1), 0,             cam_params(3); ...
%                   0,             cam_params(2), cam_params(4); ...
%                   0,             0,             1];
%         end
%
%         cam_dir = eye(3);
%         cam_dir = r * cam_dir;
%         cam_dir(:, 3) = -cam_dir(:, 3);
%
%         % Points on regular 2D grid with -1 Z coordinate, centered around
%         % center of camera sensor.
%         sz = size(depth);
%         [R,C] = ndgrid(1:sz(1), 1:sz(2));
%         centered_points = reshape(cat(3, R, C, -ones(size(R))), [], 3) - [floor(sz(1) / 2), floor(sz(2) / 2), 0];
%
%         scaling = 1 / cam_params(1);
%         transformed_points = [scaling, 0, 0; 0, scaling, 0; 0, 0, 1] * cam_dir * centered_points';
%         transformed_points = reshape(transformed_points, sz(1), sz(2), 3);
%         XYZcut = reshape(t, 1, 1, 3) + transformed_points .* depth;
%
%         % Feature upsampling
%         qimg = imread(qname);
%         sz = size(qimg);
%         sz_max = max(sz(1), sz(2));
%         Iqsize = size(sqresize(imread(qname), sz_max));
%         sz = size(XYZcut);
%         sz_max = max(sz(1), sz(2));
%         Idbsize = size(sqresize(XYZcut, sz_max));
%         tent_xq2d = at_featureupsample(tent_xq2d, this_gvresults.cnnfeat1size, Iqsize);
%         tent_xdb2d = at_featureupsample(tent_xdb2d, this_gvresults.cnnfeat2size, Idbsize);
%         % query ray
%         tent_ray2d = Kq^-1 * [tent_xq2d; ones(1, size(tent_xq2d, 2))];
%         % DB 3d points
%         indx = sub2ind(size(XYZcut(:, :, 1)), tent_xdb2d(2, :), tent_xdb2d(1, :));
%         X = XYZcut(:,:,1);
%         Y = XYZcut(:,:,2);
%         Z = XYZcut(:,:,3);
%         tent_xdb3d = [X(indx); Y(indx); Z(indx)];
%         tent_xdb3d = bsxfun(@plus, P_after(1:3, 1:3) * tent_xdb3d, P_after(1:3, 4));
%         % Select keypoint correspond to 3D
%         idx_3d = all(~isnan(tent_xdb3d), 1);
%         tent_xq2d = tent_xq2d(:, idx_3d);
%         tent_xdb2d = tent_xdb2d(:, idx_3d);
%         tent_ray2d = tent_ray2d(:, idx_3d);
%         tent_xdb3d = tent_xdb3d(:, idx_3d);
%
%
%         tentatives_2d = [tent_xq2d; tent_xdb2d];
%         tentatives_3d = [tent_ray2d; tent_xdb3d];
%
%
%         % solver
%         if size(tentatives_2d, 2) < 3
%             P = nan(3, 4);
%             inls = false(1, size(tentatives_2d, 2));
%         else
%             [ P, inls ] = ht_lo_ransac_p3p( tent_ray2d, tent_xdb3d, 1.0*pi/180);
%             if isempty(P)
%                 P = nan(3, 4);
%             end
%         end
%
%
%
%         if exist(fullfile(params.output_pnp_dense_inlier_dir, qname), 'dir') ~= 7
%             mkdir(fullfile(params.output_pnp_dense_inlier_dir, qname));
%         end
%         save('-v6', this_densepe_matname, 'P', 'inls', 'tentatives_2d', 'tentatives_3d');
%
%     %     %debug
%     %     Iq = imread(fullfile(params.data.dir, params.data.q.dir, qname));
%     %     Idb = imread(fullfile(params.data.dir, params.data.db.cutout.dir, dbname));
%     %     points.x2 = tentatives_2d(3, inls);
%     %     points.y2 = tentatives_2d(4, inls);
%     %     points.x1 = tentatives_2d(1, inls);
%     %     points.y1 = tentatives_2d(2, inls);
%     %     points.color = 'g';
%     %     points.facecolor = 'g';
%     %     points.markersize = 60;
%     %     points.linestyle = '-';
%     %     points.linewidth = 1.0;
%     %     show_matches2_vertical( Iq, Idb, points );
%     %
%     %     points.x2 = tentatives_2d(3, :);
%     %     points.y2 = tentatives_2d(4, :);
%     %     points.x1 = tentatives_2d(1, :);
%     %     points.y1 = tentatives_2d(2, :);
%     %     points.color = 'r';
%     %     points.facecolor = 'r';
%     %     points.markersize = 60;
%     %     points.linestyle = '-';
%     %     points.linewidth = 1.0;
%     %     show_matches2_vertical( Iq, Idb, points );
%     %
%     %     keyboard;
%     end
% end
%
function name = filename(pth)
    splits = strsplit(pth, filesep);
    name = splits(length(splits));
    name = name{:};
end

function [path] = absFnPath(varargin)
    % absFnPath Get valid absolute OS filesystem path of specified function name.
    % Relies on the file structure of this repository.

    % Get absolute path to folder containing this very file.
    [filepath, ~, ~] = fileparts(mfilename('fullpath'));
    % OS independent path creation.
    path = fullfile(filepath, '..', varargin{:});
end
%
% % function parfor_densePE( qname, dbnames, dbnamesId, posesFromHoloLens, firstQueryId, lastQueryId, params )
% %     % there are two exceptional situtations
% %     % 1. the actual query length can be lower than params.sequence.length, if current query is near the beginning of the overall sequence
% %     %       -> just use the smaller sequence. if length is 1, use P3P. this is handles by the caller
% %     % 2. we don't have poses from HoloLens, for some of the queries by the end of the overall sequence. This is because of a delay
% %     %       -> because the missing queries are coming from higher query Ids to smaller query Ids, and we are interested in the highest
% %     %           query Id in the sequence, we cannot use MCP at all (I checked it). So I can only use P3P on the single query.
%
% %     this_densepe_matname = fullfile(params.output_pnp_dense_inlier_dir, qname, sprintf('%d%s', dbnamesId, ".mat"));
%
% %     sequenceLength = size(dbnames,1);
% %     ind = 1:sequenceLength;
% %     useP3P = sequenceLength == 1;
%
% %     allCorrespondences2D = cell(1,sequenceLength);
% %     allCorrespondences3D = cell(1,sequenceLength);
% %     allTentatives2D = cell(1,sequenceLength);
% %     allTentatives3D = cell(1,sequenceLength);
% %     allInls = cell(1,sequenceLength);
% %     Ps = cell(1,sequenceLength);
%
% %     if exist(this_densepe_matname, 'file') ~= 2
%
% %         skipPoseEstimation = false;
% %         if any(isnan(posesFromHoloLens(:))) % exceptional situation 2.
% %             Ps(1,:) = {nan(3,4)};
% %             useP3P = true;
% %         end
% %         queriesWithLowTentatives = zeros(1,sequenceLength);
% %         for j=1:sequenceLength
% %             i = ind(j);
% %             dbname = dbnames{i};
% %             thisQueryName = sprintf('%d.jpg', firstQueryId + j - 1);
% %             %geometric verification results
% %             this_densegv_matname = fullfile(params.output_gv_dense_dir, thisQueryName, buildCutoutName(dbname, ".mat"));
% %             if exist(this_densegv_matname, 'file') ~= 2
% %                 % TODO: possible race condition?
% %                 % this function is executed in parfor and two different workers may be working on the same dbname at a time
% %                 warning('Executing parfor_denseGV within parfor_densePE. This is suspicious!');
% %                 fprintf('this_densegv_matname: %s\n', this_densegv_matname);
% %                 assert(false);
% %                 qfname = fullfile(params.input.feature.dir, params.dataset.query.dirname, [thisQueryName, ".mat"]);
% %                 cnnq = load(qfname, 'cnn');cnnq = cnnq.cnn;
% %                 parfor_denseGV( cnnq, thisQueryName, dbname, params );
% %             end
% %             this_gvresults = load(this_densegv_matname);
% %             tent_xq2d = this_gvresults.f1(:, this_gvresults.inls12(1, :));
% %             tent_xdb2d = this_gvresults.f2(:, this_gvresults.inls12(2, :));
%
%
% %             %depth information
% %             this_db_matname = fullfile(params.dataset.db.cutouts.dir, [dbname, ".mat"]);
% %             load(this_db_matname, 'XYZcut');
%
% %             %Feature upsampling
% %             Idbsize = size(XYZcut);
% %             Iqsize = Idbsize; % we padded the queries to match cutout aspect ratio (and rescaled to cutout dimensions
% %             tent_xq2d = at_featureupsample(tent_xq2d,this_gvresults.cnnfeat1size,Iqsize);
% %                 % without this, the features in query image would not match the cutout aspect ratio
% %             tent_xdb2d = at_featureupsample(tent_xdb2d,this_gvresults.cnnfeat2size,Idbsize);
% %                 % this may not be necessary
% %             %query ray
%
% %             % convert xq2d to match original query image
% %             queryWidth = params.camera.sensor.size(2);
% %             queryHeight = params.camera.sensor.size(1);
% %             cutoutWidth = Idbsize(2);
% %             cutoutHeight = Idbsize(1);
% %             tent_xq2d = adjust_inliers_to_match_original_query(tent_xq2d, queryWidth, queryHeight, cutoutWidth, cutoutHeight);
%
% %             K = params.camera.K;
%
% %             tent_ray2d = K^-1 * [tent_xq2d; ones(1, size(tent_xq2d, 2))];
% %             %DB 3d points
% %             indx = sub2ind(size(XYZcut(:,:,1)),tent_xdb2d(2,:),tent_xdb2d(1,:));
% %             X = XYZcut(:,:,1);Y = XYZcut(:,:,2);Z = XYZcut(:,:,3);
% %             tent_xdb3d = [X(indx); Y(indx); Z(indx)];
% %             %Select keypoint correspond to 3D
% %             idx_3d = all(~isnan(tent_xdb3d), 1); % this typically contains only one
% %             tent_xq2d = tent_xq2d(:, idx_3d);
% %             tent_xdb2d = tent_xdb2d(:, idx_3d);
% %             tent_ray2d = tent_ray2d(:, idx_3d);
% %             tent_xdb3d = tent_xdb3d(:, idx_3d);
% %             allCorrespondences2D{i} = tent_xq2d;
% %             allCorrespondences3D{i} = tent_xdb3d;
%
% %             tentatives_2d = [tent_xq2d; tent_xdb2d];
% %             tentatives_3d = [tent_ray2d; tent_xdb3d];
% %             allTentatives2D{i} = tentatives_2d;
% %             allTentatives3D{i} = tentatives_3d;
% %             allInls{i} = ones(1,size(tentatives_2d,2));
%
% %             if size(tentatives_2d, 2) < 3
% %                 queriesWithLowTentatives(j) = true;
% %             end
% %         end
% %         nQueriesWithoutLowTentatives = sequenceLength-sum(queriesWithLowTentatives);
% %         if queriesWithLowTentatives(end)
% %             Ps(1,:) = {nan(3,4)};
% %             skipPoseEstimation = true;
% %         elseif nQueriesWithoutLowTentatives < 2
% %             Ps(1,:) = {nan(3,4)};
% %             useP3P = true;
% %             % TODO: Should I really use P3P in this case, or do MCP which skips the broken queries and replaces them with those
% %             % for which we have inliers? This might be a future improvement
% %         end
%
% %         if ~skipPoseEstimation
% %             if useP3P
% %                 %solver
% %                 [ P, inls ] = ht_lo_ransac_p3p( tent_ray2d, tent_xdb3d, 1.0*pi/180);
% %                 if isempty(P)
% %                     P = nan(3, 4);
% %                 end
% %                 Ps{end} = P;
% %                 allInls{end} = inls;
% %             else
% %                 workingDir = tempname;
% %                 %workingDir = '/Volumes/GoogleDrive/Můj disk/ARTwin/InLocCIIRC_dataset/evaluation/sequences'; % only for debugging;
% %                 %                                                                                                % TODO: use better path;
% %                 %                                                                                                % TODO: remove
% %                 %                                                                                                % this does NOT support multiple experiments
% %                 inlierThreshold = 12.0; % TODO
% %                 numLoSteps = 10; % TODO; why is this parameter seem to have no effect (I tried 0, 1, 10, 100).
% %                                 % It is actualy correctly used in RansacLib: ransac.h:378...
% %                 invertYZ = false; % TODO
% %                 pointsCentered = false;
% %                 undistortionNeeded = false; % TODO
% %                 queryInd = [firstQueryId:lastQueryId]';
% %                 % what happens if some 2D correspondences are outside the camera rectangle? Nothing bad (looking at MCP code)!
% %                 Ps = multiCameraPose(workingDir, queryInd, posesFromHoloLens, ...
% %                                                     allCorrespondences2D, allCorrespondences3D, ...
% %                                                     inlierThreshold, numLoSteps, ...
% %                                                     invertYZ, pointsCentered, undistortionNeeded, ...
% %                                                     queryWidth, queryHeight, K, ...
% %                                                     params); % wrt model
% %                 %fprintf('Not deleting MCP working dir: %s\n', workingDir);
% %                 rmdir(workingDir, 's');
% %             end
% %         end
%
%
% %         % TODO: possible race condition?
% %         % this function is executed in parfor and two different workers may be working on the same qname at a time
% %         % if that happens, though, mkdir is noop but it shows an error.
% %         if exist(fullfile(params.output_pnp_dense_inlier_dir, qname), 'dir') ~= 7
% %             mkdir(fullfile(params.output_pnp_dense_inlier_dir, qname));
% %         end
% %         % how to compute inls using MCP? The inls are not easily extractable from MCP, so don't do it for now.
% %         % If I dont save inls, then queryPipeline will break. Yes, I could just
% %         % use all tentative inliers there instead. But first, how are inls actually computed in p3p? inls in p3p is related to
% %         % points which reproject with a high error (above cos(1 [deg])) - WTF??
%
% %         % for now if a) p3p used -> use inls from p3p b) if MCP used, use ones
% %         save('-v6', this_densepe_matname, 'Ps', 'allInls', 'allTentatives2D', 'allTentatives3D');
%
% %         % UNMAINTAINED LEGACY CODE
% %     %      %% debug
% %     %      qname = '1.jpg';
% %     %      dbname = 'B-315/3/cutout_3_-120_0.jpg';
% %     %      dbname = 'B-315/1/cutout_1_-60_0.jpg';
% %     %      inlierPath = '/Volumes/GoogleDrive/Můj disk/ARTwin/InLocCIIRC_dataset/outputs/PnP_dense_inlier/1.jpg/cutout_B-315_3 -120 0.pnp_dense_inlier.mat';
% %     %      inlierPath = '/Volumes/Elements/backup/1-4-2020/InLocCIIRC_dataset/outputs/PnP_dense_inlier/1.jpg/cutout_1_-60_0.pnp_dense_inlier.mat';
% %     %      load(inlierPath, 'P', 'inls', 'tentatives_2d', 'tentatives_3d');
% %     %      Iq = imread(fullfile(params.dataset.query.dir, qname));
% %     %      Idb = imread(fullfile(params.dataset.db.cutouts.dir, dbname));
% %     %      points.x2 = tentatives_2d(3, inls);
% %     %      points.y2 = tentatives_2d(4, inls);
% %     %      points.x1 = tentatives_2d(1, inls);
% %     %      points.y1 = tentatives_2d(2, inls);
% %     %      points.color = 'g';
% %     %      points.facecolor = 'g';
% %     %      points.markersize = 60;
% %     %      points.linestyle = '-';
% %     %      points.linewidth = 1.0;
% %     %      show_matches2_vertical( Iq, Idb, points );
% %     %
% %     %      points.x2 = tentatives_2d(3, :);
% %     %      points.y2 = tentatives_2d(4, :);
% %     %      points.x1 = tentatives_2d(1, :);
% %     %      points.y1 = tentatives_2d(2, :);
% %     %      points.color = 'r';
% %     %      points.facecolor = 'r';
% %     %      points.markersize = 60;
% %     %      points.linestyle = '-';
% %     %      points.linewidth = 1.0;
% %     %      show_matches2_vertical( Iq, Idb, points );
% %     %
% %     %      keyboard;
% %     end
% % end
