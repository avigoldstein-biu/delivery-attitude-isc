% Direct surface painting: sample AAL atlas at each surface vertex, colour by stat.
set(0,'DefaultFigureVisible','off');
addpath('D:/fieldtrip'); ft_defaults; warning('off','all');

atlas = ft_read_atlas('D:/fieldtrip/template/atlas/aal/ROI_MNI_V4.nii');   % mm space
T = atlas.transform; dim = atlas.dim; Tinv = inv(T);
tissue = zeros(dim); n = 1;
for a = [1:20 23:36 43:70 81:90]; tissue(atlas.tissue==a) = n; n = n+1; end

S = load('G:/Barak1/reanalysis/render_stats.mat');
panels = {'s1_beta_raw','s1_beta_res','s1_beta_core','s2_alpha','s2_theta'};
cmap = jet(256); CLIM = [0 0.09]; grey = [0.86 0.83 0.73];
outdir = 'G:/Barak1/reanalysis/renders/'; if ~exist(outdir,'dir'); mkdir(outdir); end
hemis = {'left','right'}; azlat = [-90 90]; azmed = [90 -90];

% preload + pre-sample surfaces (vertex -> region) once
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
  fprintf('%s surface: %d verts, %d in a region\n', hemis{hi}, size(pos,1), sum(reg>0));
end

for pi = 1:numel(panels)
  p = S.(panels{pi}); val72 = nan(72,1);
  for i = 1:72; if p.mask(i)>0; val72(i) = max(p.val(i),0); end; end
  for hi = 1:2
    reg = surf(hi).reg; nv = numel(reg);
    vals = nan(nv,1); good = reg>=1 & reg<=72;
    vals(good) = val72(reg(good)); vok = ~isnan(vals);
    C = repmat(grey, nv, 1);
    ci = round(1 + 255*(vals(vok)-CLIM(1))/(CLIM(2)-CLIM(1))); ci = min(max(ci,1),256);
    C(vok,:) = cmap(ci,:);
    for view_i = 1:2   % 1 = lateral, 2 = medial
      az = (view_i==1)*azlat(hi) + (view_i==2)*azmed(hi);
      f = figure('Color','w','Position',[100 100 700 550]);
      patch('Vertices',surf(hi).pos,'Faces',surf(hi).tri,'FaceVertexCData',C, ...
            'FaceColor','interp','EdgeColor','none','FaceLighting','gouraud');
      axis equal off vis3d; view(az,0); camlight('headlight'); material dull;
      nm = [hemis{hi}(1) ternary(view_i==1,'lat','med')];   % Llat/Lmed/Rlat/Rmed style
      nm = [upper(hemis{hi}(1)) ternary(view_i==1,'lat','med')];
      out = [outdir panels{pi} '_' nm '.png'];
      print(f,out,'-dpng','-r150'); close(f);
      fprintf('SAVED %s\n', out);
    end
  end
end
disp('PATCH_RENDERS_DONE');

function r = ternary(c,a,b); if c; r=a; else; r=b; end; end
