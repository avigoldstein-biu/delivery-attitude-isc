% Surface render for the convergence figure (Figure 6, panel A).
%
% Three effect maps on one common zero-based colour scale, each at its reported control stage,
% with the four parcels of the a priori right auditory-perisylvian ROI outlined on every view.
% Maps are unthresholded: the figure's claim is that two manipulations land on the same patch
% and a third does not, which a common scale shows directly.
%
% The outline is drawn from the mesh itself -- every triangle edge whose two vertices fall on
% opposite sides of the ROI is plotted -- so it follows the parcellation exactly rather than
% approximating it. Note that the insula is buried in the sylvian fissure and so contributes
% almost no visible surface; the outline on a lateral view effectively shows the other three
% parcels.
%
% Reads  <DERIV>/convergence_maps.mat   (fig_convergence_export.py)
% Writes <DERIV>/renders/conv_*_{Llat,Lmed,Rlat,Rmed}.png
set(0,'DefaultFigureVisible','off');
addpath('D:/fieldtrip'); ft_defaults; warning('off','all');

DERIV = 'G:/Barak1/reanalysis';
atlas = ft_read_atlas('D:/fieldtrip/template/atlas/aal/ROI_MNI_V4.nii');
T = atlas.transform; dim = atlas.dim; Tinv = inv(T);
tissue = zeros(dim); n = 1;
for a = [1:20 23:36 43:70 81:90]; tissue(atlas.tissue==a) = n; n = n+1; end

S = load([DERIV '/convergence_maps.mat']);
panels = {'conv_s1_delivery','conv_s2_attitude','conv_s3_cue'};
roi_regions = find(S.roi(:) > 0);
% Same colour convention as Figures 3 and 4 (render_patch.m): sequential, zero-based, negative
% values clipped to zero. Every test behind these maps is one-sided, so a diverging scale would
% imply bidirectional inference the paper does not make, and matching CLIM lets a reader compare
% this figure with Figures 3 and 4 directly.
CLIM = [0 0.09];
% Grey-to-red sequential ramp: lightness falls monotonically with effect size, so the scale
% reads in one direction, matching the one-sided tests. Vertices belonging to no atlas parcel
% are drawn in a lighter cream than the ramp's zero end, so "no parcel" cannot be mistaken for
% "zero effect".
cpts = [0.78 0.78 0.78; 0.93 0.45 0.32; 0.55 0.00 0.00];
tt = [0; 0.5; 1]; tq = linspace(0,1,256)';
cmap = [interp1(tt,cpts(:,1),tq) interp1(tt,cpts(:,2),tq) interp1(tt,cpts(:,3),tq)];
grey = [0.86 0.86 0.86];   % vertices in no parcel: close to the ramp's zero so they do
                           % not read as holes, but light enough to remain distinguishable

outdir = [DERIV '/renders/']; if ~exist(outdir,'dir'); mkdir(outdir); end
hemis = {'left','right'}; azlat = [-90 90]; azmed = [90 -90];

surf = struct();
for hi = 1:2
  L = load(['D:/fieldtrip/template/anatomy/surface_white_' hemis{hi} '.mat']);
  fn = fieldnames(L); msh = L.(fn{1});
  if isfield(msh,'pos'); pos = msh.pos; else; pos = msh.pnt; end
  tri = msh.tri;
  ijk = round((Tinv(1:3,:) * [pos'; ones(1,size(pos,1))])');
  ok = all(ijk>=1,2) & ijk(:,1)<=dim(1) & ijk(:,2)<=dim(2) & ijk(:,3)<=dim(3);
  reg = zeros(size(pos,1),1);
  reg(ok) = tissue(sub2ind(dim, ijk(ok,1), ijk(ok,2), ijk(ok,3)));
  surf(hi).pos = pos; surf(hi).tri = tri; surf(hi).reg = reg;

  % ROI boundary: triangle edges straddling the ROI
  inroi = ismember(reg, roi_regions);
  E = [tri(:,[1 2]); tri(:,[2 3]); tri(:,[3 1])];
  E = unique(sort(E,2),'rows');
  bnd = E(inroi(E(:,1)) ~= inroi(E(:,2)), :);
  surf(hi).bnd = bnd;
  fprintf('%s: %d verts, %d in ROI, %d boundary edges\n', ...
          hemis{hi}, size(pos,1), sum(inroi), size(bnd,1));
end

for pi = 1:numel(panels)
  val72 = S.(panels{pi}); val72 = val72(:);
  for hi = 1:2
    reg = surf(hi).reg; nv = numel(reg);
    vals = nan(nv,1); good = reg>=1 & reg<=72;
    vals(good) = max(val72(reg(good)), 0);          % one-sided: clip negatives, as in render_patch.m
    vok = ~isnan(vals);
    C = repmat(grey, nv, 1);
    ci = round(1 + (size(cmap,1)-1)*(vals(vok)-CLIM(1))/(CLIM(2)-CLIM(1)));
    ci = min(max(ci,1),size(cmap,1));
    C(vok,:) = cmap(ci,:);
    P = surf(hi).pos; B = surf(hi).bnd;
    % nudge the outline outward along the vertex normal direction so it is not hidden by the mesh
    ctr = mean(P,1); dirs = P - ctr; dirs = dirs ./ (vecnorm(dirs,2,2)+eps);
    Pout = P + 0.6*dirs;
    for view_i = 1:2
      az = (view_i==1)*azlat(hi) + (view_i==2)*azmed(hi);
      f = figure('Color','w','Position',[100 100 700 550]);
      patch('Vertices',P,'Faces',surf(hi).tri,'FaceVertexCData',C, ...
            'FaceColor','interp','EdgeColor','none','FaceLighting','gouraud');
      hold on;
      if ~isempty(B)
        X = [Pout(B(:,1),1) Pout(B(:,2),1)]'; Y = [Pout(B(:,1),2) Pout(B(:,2),2)]';
        Z = [Pout(B(:,1),3) Pout(B(:,2),3)]';
        line(X, Y, Z, 'Color','k', 'LineWidth',1.1);
      end
      axis equal off vis3d; view(az,0); camlight('headlight'); material dull;
      nm = [upper(hemis{hi}(1)) ternary(view_i==1,'lat','med')];
      out = [outdir panels{pi} '_' nm '.png'];
      print(f,out,'-dpng','-r150'); close(f);
      fprintf('SAVED %s\n', out);
    end
  end
end
disp('CONV_RENDERS_DONE');

function r = ternary(c,a,b); if c; r=a; else; r=b; end; end
