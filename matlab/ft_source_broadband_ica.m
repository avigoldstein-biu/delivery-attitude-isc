% Broadband source reconstruction WITH the objective ocular ICA, for the spectral-power
% and specparam analyses (Section 2.3.1 / Fig 2a).
%
% The ICA is not recomputed independently: ft_ica_full3.m seeds runica with rng(1000+si)
% and built the decomposition from all three conditions resampled to 300 Hz and
% concatenated. Repeating that here with the same seed and the same input reproduces the
% SAME components and the same rejection list, so the band data (source_myica3/) and the
% broadband data carry an identical ocular correction. The script checks this against the
% n_rejected value stored by ft_ica_full3.m and warns if they diverge.
%
% Output: reanalysis/source_bb_ica/<subj>.mat
%           Charismatic / Non_Charismatic / Silent = [72 x T@250 Hz], plus n_rejected.
%
% Set NSUB to limit the number of subjects when testing:  setenv('NSUB','2')
addpath('D:/fieldtrip'); ft_defaults;
global ft_default
ft_default.showcallinfo='no'; ft_default.trackcallinfo='no';
warning('off','all');
ROOT='G:/Barak1/Charisma'; OUT='G:/Barak1/reanalysis/source_bb_ica';
BAND='G:/Barak1/reanalysis/source_myica3';                  % for the n_rejected cross-check
if ~exist(OUT,'dir'), mkdir(OUT); end
SFM=1017.25; FSD=250;
CODES=[220 240 250]; CONDS={'Charismatic','Non_Charismatic','Silent'}; DUR=[150.24 140.16 153.60];
atl=load('G:/Barak1/Charisma/atlas_info.mat'); NREG=numel(atl.tissuelabel);
subs=dir(fullfile(ROOT,'char_*')); [~,o]=sort(cellfun(@(x)str2double(x(6:end)),{subs.name})); subs=subs(o);
lim=getenv('NSUB'); if ~isempty(lim), subs=subs(1:str2double(lim)); end
ndone=numel(dir(fullfile(OUT,'char_*.mat'))); ttot=0; nrun=0; mismatch={};
fprintf('=== %d of %d subjects already complete ===\n',ndone,numel(subs));

for si=1:numel(subs)
  t0=tic;
  tag=subs(si).name; sd=fullfile(ROOT,tag); outf=fullfile(OUT,[tag '.mat']);
  if exist(outf,'file'), fprintf('%s skip\n',tag); continue; end
  rng(1000+si);                                  % same seed as ft_ica_full3.m
  megf=dir(fullfile(sd,'*lf_c,rfhp0.1Hz')); megf=fullfile(sd,megf(1).name);
  hm=load(fullfile(sd,'headmodelN.mat')); headmodel=ft_convert_units(hm.headmodelN,'cm');
  sm=load(fullfile(sd,'sourcemodel.mat')); sourcemodel=sm.sourcemodel;
  sourcemodel.inside=logical(sourcemodel.inside(:)); sourcemodel=ft_convert_units(sourcemodel,'cm');
  ad=load(fullfile(sd,'audiodata.mat')); ad=ad.audiodata;
  tissue=atl.tissue(:); tissue=tissue(sourcemodel.inside);

  % ---- read every available condition at full rate (same order as ft_ica_full3.m) ----
  clip={}; have=[];
  for ci=1:numel(CODES)
    idx=find(ad.trialinfo==CODES(ci),1);
    if isempty(idx), fprintf('%s: no %s trial\n',tag,CONDS{ci}); continue; end
    onset=ad.sampleinfo(idx,1)-round(1.30*SFM); ns=round(DUR(ci)*SFM);
    cfg=[]; cfg.dataset=megf; cfg.trl=[onset onset+ns-1 0]; cfg.channel='MEG';
    clip{end+1}=ft_preprocessing(cfg); have(end+1)=ci; %#ok<SAGROW>
  end
  if numel(have)<2, fprintf('%s: fewer than 2 conditions, skipping\n',tag); continue; end

  % ---- ICA, identical construction to ft_ica_full3.m ----
  rs={}; for k=1:numel(clip), cfg=[]; cfg.feedback='none'; cfg.resamplefs=300; rs{k}=ft_resampledata(cfg,clip{k}); end
  cfg=[]; cat_=ft_appenddata(cfg,rs{:}); clear rs
  cfg=[]; cfg.method='runica'; cfg.numcomponent=40; cfg.runica.maxsteps=120;
  cfg.runica.verbose='off'; cfg.feedback='none'; comp=ft_componentanalysis(cfg,cat_); clear cat_
  [~,loc]=ismember(comp.topolabel,clip{1}.grad.label); pos=clip{1}.grad.chanpos(loc,:);
  ant=pos(:,2)>prctile(pos(:,2),70) & pos(:,3)<prctile(pos(:,3),50);
  reject=[];
  for c=1:size(comp.topo,2)
    t=comp.topo(:,c); front=norm(t(ant))/norm(t); x=comp.trial{1}(c,:); k=kurtosis(x);
    P=abs(fft(x-mean(x))).^2; f=(0:numel(x)-1)/numel(x)*300; lfrac=sum(P(f>=1&f<4))/sum(P(f>=1&f<40));
    if front>0.45 && lfrac>0.4 && k>5, reject(end+1)=c; end %#ok<SAGROW>
  end
  % cross-check against the band-data run
  bf=fullfile(BAND,[tag '.mat']);
  if exist(bf,'file')
    b=load(bf,'n_rejected');
    if isfield(b,'n_rejected') && b.n_rejected~=numel(reject)
      fprintf(2,'%s: WARNING ICA differs from band run (%d vs %d components)\n',tag,b.n_rejected,numel(reject));
      mismatch{end+1}=tag; %#ok<SAGROW>
    end
  end
  fprintf('%s: reject %d comps %s\n',tag,numel(reject),mat2str(reject));

  % ---- clean, broadband filter, beamform, downsample ----
  lf=[]; reg=struct();
  for k=1:numel(clip)
    ci=have(k);
    X=clip{k}.trial{1}; [~,order]=ismember(clip{k}.label,comp.topolabel);
    topo=comp.topo(order,:); unmix=comp.unmixing(:,order);
    if isempty(reject), Xc=X; else Xd=X-mean(X,2); S=unmix*Xd; Xc=X-topo(:,reject)*S(reject,:); end
    cln=clip{k}; cln.trial{1}=Xc;
    cfg=[]; cfg.bpfilter='yes'; cfg.bpfreq=[1 40]; cfg.bpfilttype='firws'; cfg.feedback='none';
    db=ft_preprocessing(cfg,cln);
    if isempty(lf)
      cfg=[]; cfg.sourcemodel=sourcemodel; cfg.headmodel=headmodel; cfg.channel='MEG'; cfg.feedback='none';
      lf=ft_prepare_leadfield(cfg,db);
    end
    cfg=[]; cfg.covariance='yes'; cfg.covariancewindow='all'; tl=ft_timelockanalysis(cfg,db);
    cfg=[]; cfg.method='lcmv'; cfg.sourcemodel=lf; cfg.headmodel=headmodel;
    cfg.lcmv.lambda='10%'; cfg.lcmv.keepfilter='yes'; cfg.lcmv.fixedori='yes'; cfg.lcmv.feedback='none';
    src=ft_sourceanalysis(cfg,tl);
    F=cat(1,src.avg.filter{sourcemodel.inside}); vts=F*db.trial{1};
    ns=size(db.trial{1},2); R=zeros(NREG,ns,'single');
    for r=1:NREG, v=vts(tissue==r,:); if ~isempty(v), R(r,:)=single(mean(v,1)); end, end
    Rd=resample(double(R'),FSD,round(SFM))';                 % downsample to 250 Hz
    reg.(CONDS{ci})=single(Rd);
  end
  reg.n_rejected=numel(reject);
  save(outf,'-struct','reg','-v7');
  el=toc(t0)/60; nrun=nrun+1; ttot=ttot+el; ndone=ndone+1;
  left=(numel(subs)-ndone)*(ttot/nrun);
  fprintf('[%2d/%d] %s  %.1f min  | this session %.0f min, ETA %.0f min (%.1f h)\n', ...
          ndone,numel(subs),tag,el,ttot,left,left/60);
end
if isempty(mismatch), fprintf('ICA reproduced the band run for every subject\n');
else, fprintf(2,'ICA DIFFERED for: %s\n',strjoin(mismatch,', ')); end
fprintf('ALL DONE\n');
