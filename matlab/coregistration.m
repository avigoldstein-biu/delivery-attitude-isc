%% co-registration, head and source model per subject 
% An individual analysis for each subject

% Headshape
hs=ft_read_headshape('xc,hb,lf_c,rfhp0.1Hz'); %subject’s clean head shape file from digitization
% plot data to detect outlier dots - remove if needed
scatter3(hs.pos(:,1),hs.pos(:,2),hs.pos(:,3))

%% realign

load ('mri.mat')
hs.coordsys='4d'; % our MEG cor system 
cfg=[]; 
cfg.method= 'headshape'; 
cfg.coordsys='4d';
cfg.headshape.icp='yes'; % automatic realignment 
cfg.headshape.interactive='yes';
cfg.headshape.headshape=hs;
cfg.viewresult='yes'; % ‘yes’ will allow us to see MRI after realign 
mri_realigned=ft_volumerealign(cfg,mri);

cfg=[]; 
cfg.method= 'headshape'; 
cfg.coordsys='4d';
cfg.headshape.icp='no'; % automatic realignment 
cfg.headshape.interactive='yes';
cfg.headshape.headshape=hs;
cfg.viewresult='yes'; 
mri_realigned=ft_volumerealign(cfg,mri_realigned);

% change units and save
mri_realigned = ft_convert_units(mri_realigned,'cm');
save mri_realigned mri_realigned

%% check how it looks and change manually if needed
cfg = [];
cfg.output = {'brain'}% 'skull' 'scalp'};
mri_segmented = ft_volumesegment(cfg, mri_realigned);

% create mesh
cfg = [];
cfg.method = 'projectmesh';
cfg.tissue = 'brain';
cfg.numvertices = 3000;
mesh_brain = ft_prepare_mesh(cfg, mri_segmented);

cfg = [];
cfg.method = 'singleshell';
headmodel = ft_prepare_headmodel(cfg, mesh_brain);
headmodelN = ft_convert_units(headmodel,'cm');

save('headmodelN', 'headmodelN')

% source model with template grid

[ftver, ftpath] = ft_version;
% load MRI template ('vol')
load(fullfile(ftpath, 'template/headmodel/standard_singleshell.mat'));

cfg = [];
cfg.xgrid  = -20:1:20;
cfg.ygrid  = -20:1:20;
cfg.zgrid  = -20:1:20;
cfg.unit   = 'cm';
cfg.tight  = 'yes';
cfg.inwardshift = -1.5;
cfg.headmodel   = vol;
template_grid   = ft_prepare_sourcemodel(cfg);
template_grid = ft_convert_units(template_grid,'cm');
%template_grid.coordsys = 'mni';

% % see headshape 
figure;
hold on
ft_plot_mesh(template_grid.pos(template_grid.inside,:));
ft_plot_headmodel(vol,  'facecolor', 'cortex', 'edgecolor', 'none');
ft_plot_axes(vol);
alpha 0.5
camlight

% read atlas 
atlas = ft_read_atlas(fullfile(ftpath, 'template/atlas/aal/ROI_MNI_V4.nii'));
atlas = ft_convert_units(atlas,'cm');
cfg = [];
cfg.atlas      = atlas;
cfg.roi = atlas.tissuelabel;
cfg.inputcoord = 'mni';
mask           = ft_volumelookup(cfg, template_grid);
template_grid.inside = false(template_grid.dim); 
template_grid.inside(mask==1) = true;

figure;
ft_plot_mesh(template_grid.pos(template_grid.inside,:));

cfg                = [];
cfg.warpmni   = 'yes';
cfg.template  = template_grid;
cfg.nonlinear = 'yes'; 
cfg.mri            = mri_realigned;
sourcemodelN        = ft_prepare_sourcemodel(cfg);
sourcemodel = sourcemodelN;
save('sourcemodel', 'sourcemodel');